"""Real-Time Chunking for MolmoAct2 (Studio commit 8f13241 has none): export-time patch + runtime wrapper.

Export side (`patch_policy(policy)` before `policy.export(...)`, see export.py --rtc):
  MolmoAct2Model.predict_action_chunk is replaced by a version that, when the batch carries
  `prev_chunk_left_over` (B, chunk, 6) in RUNTIME units (our LeRobot degrees / gripper 0-100), `inference_delay`,
  `execution_horizon` (int64 scalars) and `max_guidance_weight` (float32 scalar), steers the Euler flow integration
  toward the unexecuted tail of the previous chunk (Studio's training-free RTC guidance, RTCModelMixin._rtc_correct,
  re-derived for MolmoAct2's time direction):
    * MolmoAct2 integrates t = 0 (zeros/noise) -> 1 (clean) with x += dt * v, so the clean-action estimate at t is
      x1 = x_t + (1 - t) * v, and the pull toward the previous chunk enters as v += w(t) * W * (prev - x1).
    * w(t) uses the same schedule as Studio with tau = t: tau = 0 -> max weight, tau -> 1 -> 0.
    * W = linear prefix weights (1 up to the inference delay, ramping to 0 at the execution horizon).
  The previous chunk is mapped into the model space inside the graph: checkpoint frame (sign * x + offset, unit
  scale) then quantile normalisation with the exported action statistics, zero-padded to max_action_dim.
  With max_guidance_weight ~ 0 the output equals the plain export.
Runtime side (`MolmoRTCPolicy`): Studio's runtime MolmoAct2 input builder drops unknown keys, so the wrapper runs the
  manifest preprocessors, adds the four RTC inputs, then calls the adapter and postprocessors itself. It has the same
  call signature / `rtc_next` hand-off as deploy/pi05/rtc_control.py, so the RTC controller works unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RTC_KEYS = ("prev_chunk_left_over", "inference_delay", "execution_horizon", "max_guidance_weight")
SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------- export side
def _action_quantiles(policy):
    from physicalai.data.observation import FeatureType
    from physicalai.policies.utils.features import get_feature_by_type

    feat = get_feature_by_type(policy.output_features, FeatureType.ACTION)
    nd = feat.normalization_data
    q01 = np.asarray(nd.q01, np.float32).reshape(-1)
    q99 = np.asarray(nd.q99, np.float32).reshape(-1)
    return q01, q99


def patch_policy(policy) -> dict:
    """Patch policy.model (MolmoAct2Model) for RTC and return the RTC export input sample entries."""
    import torch

    model = policy.model
    bb = model.backbone.model
    chunk = int(model._chunk_size)
    q01, q99 = _action_quantiles(policy)
    adim = len(q01)
    signs = torch.tensor(SIGNS[:adim], dtype=torch.float32)
    offsets = torch.tensor(OFFSETS[:adim], dtype=torch.float32)
    q01_t = torch.tensor(q01)
    den_t = torch.tensor(np.where(q99 - q01 == 0, 1e-8, q99 - q01).astype(np.float32))
    orig = model.predict_action_chunk

    def to_model_space(prev):  # (B, chunk, adim) runtime units -> normalized, padded to max_action_dim
        ck = prev * signs.to(prev) + offsets.to(prev)
        n = 2.0 * (ck - q01_t.to(prev)) / den_t.to(prev) - 1.0
        n = torch.clamp(n, -1.0, 1.0)
        pad = bb.max_action_dim - n.shape[-1]
        return torch.nn.functional.pad(n, (0, pad)) if pad > 0 else n

    def prefix_weights(delay, horizon_t, device):
        end = horizon_t.float().reshape(())
        start = torch.minimum(delay.float().reshape(()), end)
        idx = torch.arange(chunk, dtype=torch.float32, device=device)
        w = torch.clamp((end - idx) / (end - start + 1.0), 0.0, 1.0)
        return w.reshape(1, chunk, 1)

    @torch.no_grad()
    def predict_action_chunk(batch, *, sample_noise=None, generator=None):
        if "prev_chunk_left_over" not in batch:
            return orig(batch, sample_noise=sample_noise, generator=generator)
        from physicalai.policies.molmoact2.model import _MODEL_INPUT_KEYS

        inputs = {k: batch[k] for k in _MODEL_INPUT_KEYS if k in batch}
        ae = bb._require_action_expert()
        steps = int(bb.flow_matching_num_steps)
        ids = inputs["input_ids"]
        B = ids.shape[0]
        device = ae.action_embed.weight.device
        dtype = ae.action_embed.weight.dtype
        context = bb._encode_action_context(
            input_ids=ids, attention_mask=inputs.get("attention_mask"), token_type_ids=inputs.get("token_type_ids"),
            images=inputs.get("images"), token_pooling=inputs.get("token_pooling"), seq_len=chunk, device=device,
            dtype=dtype)
        pad_mask = inputs.get("action_dim_is_pad")
        prev = to_model_space(batch["prev_chunk_left_over"].to(torch.float32)).to(dtype)
        w = prefix_weights(batch["inference_delay"], batch["execution_horizon"], device).to(dtype)
        gmax = batch["max_guidance_weight"].to(torch.float32).reshape(())
        # start exactly like the original loop (generate_actions_from_inputs): sampled noise when the policy uses
        # random input noise (traced as the same fixed-seed op as the plain export), zeros otherwise
        use_noise = model._use_random_input_noise if sample_noise is None else sample_noise
        shape = (B, chunk, bb.max_action_dim)
        if use_noise:
            traj = bb._mask_action_dims(torch.randn(*shape, device=device, dtype=dtype, generator=generator), pad_mask)
        else:
            traj = torch.zeros(*shape, device=device, dtype=dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = step / steps
            ts = torch.full((B,), t, device=device, dtype=dtype)
            v = bb._mask_action_dims(ae.forward_with_context(traj, ts, context=context), pad_mask)
            x1 = traj + (1.0 - t) * v
            err = (prev - x1) * w
            if t == 0.0:
                gw = gmax
            else:
                c = (1.0 - t) / t
                inv_r2 = ((1.0 - t) ** 2 + t ** 2) / ((1.0 - t) ** 2)
                gw = torch.minimum(torch.tensor(c * inv_r2, dtype=torch.float32), gmax)
            v = v + gw.to(dtype) * err
            traj = bb._mask_action_dims(traj + dt * v, pad_mask)
        return traj[:, :chunk, : model._action_dim].float()

    model.predict_action_chunk = predict_action_chunk
    return {
        "prev_chunk_left_over": torch.zeros(1, chunk, adim, dtype=torch.float32),
        "inference_delay": torch.tensor(0, dtype=torch.int64),
        "execution_horizon": torch.tensor(1, dtype=torch.int64),
        "max_guidance_weight": torch.tensor(1e-8, dtype=torch.float32),
    }


# ----------------------------------------------------------------------------- runtime side
def make_molmo_rtc_policy_class(base):
    """Subclass of dlsj.deploy.pi05_runtime.Pi05Policy for an RTC-patched MolmoAct2 export."""
    sys.path.insert(0, str(Path("~/inference/deploy/pi05").expanduser()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05"))
    from rtc_control import rtc_inputs

    class MolmoRTCPolicy(base):
        def __init__(self, *a, horizon: int = 15, guidance: float = 3.0, chunk: int = 30, **kw):
            super().__init__(*a, **kw)
            names = set(self.model.adapter.input_names or [])
            # plain export: behave like the base policy (no second model load); the controller then crossfades
            self.rtc_enabled = "prev_chunk_left_over" in names
            self.horizon, self.guidance, self.chunk = horizon, guidance, chunk
            self.rtc_next = None
            self.last_rtc: dict = {}

        def __call__(self, state, images, task):
            if not self.rtc_enabled:
                self.rtc_next = None
                return super().__call__(state, images, task)
            nxt, self.rtc_next = self.rtc_next, None
            left, delay = nxt if nxt is not None else (None, 0)
            obs = {"state": np.asarray(state, np.float32)[None], "task": [task]}
            for cam in self.cameras:
                obs[f"images.{cam}"] = np.ascontiguousarray(np.asarray(images[cam], np.uint8).transpose(2, 0, 1)[None])
            extra = rtc_inputs(left, delay, self.horizon, self.guidance, self.chunk, len(state))
            self.last_rtc = {"tail": 0 if left is None else len(left), "delay": int(extra["inference_delay"]),
                             "guidance": float(extra["max_guidance_weight"])}
            m = self.model
            x = dict(obs)
            for p in m.preprocessors:
                x = p(x)
            x.update({k: np.asarray(v) for k, v in extra.items()})
            out = m.runner.run(m.adapter, m._prepare_inputs(x))
            for p in m.postprocessors:
                out = p(out)
            a = np.asarray(out["action"])
            return a[0] if a.ndim == 3 else a

    return MolmoRTCPolicy


def guidance_weight_schedule(steps: int, gmax: float) -> list[float]:
    """For docs/tests: the per-step guidance weight the patched loop applies."""
    out = []
    for s in range(steps):
        t = s / steps
        if t == 0:
            out.append(gmax)
        else:
            out.append(min(((1 - t) / t) * (((1 - t) ** 2 + t ** 2) / ((1 - t) ** 2)), gmax))
    return out


if __name__ == "__main__":
    print("guidance schedule (10 steps, gmax 3):", [round(x, 3) for x in guidance_weight_schedule(10, 3.0)])
    print("pi05-equivalent check: at t=0.5 ->", round(math.inf if False else guidance_weight_schedule(2, 99)[1], 3))


def fix_rtc_input_names(xml_path) -> list[str]:
    """Tracing leaves alias tensor names on the RTC graph inputs (e.g. {'3827', 'prev_chunk_left_over', 'prev.1'})
    and the runtime takes `any_name`, which may be an alias. Rewrite those Parameter ports in the IR XML to carry only
    the canonical name (no weights are touched). Returns the inputs that were renamed."""
    import re

    path = Path(xml_path)
    text = path.read_text()
    fixed = []

    def repl(m):
        names = [n.strip() for n in m.group(2).split(",")]
        hit = [k for k in RTC_KEYS if k in names]
        if hit and names != [hit[0]]:
            fixed.append(hit[0])
            return f'{m.group(1)}"{hit[0]}"'
        return m.group(0)

    # only Parameter layers' output ports: <layer ... type="Parameter" ...> ... <port ... names="...">
    def per_layer(m):
        return re.sub(r'(<port [^>]*names=)"([^"]*)"', repl, m.group(0))

    text2 = re.sub(r'<layer [^>]*type="Parameter"[^>]*>.*?</layer>', per_layer, text, flags=re.S)
    if text2 != text:
        path.write_text(text2)
    return fixed
