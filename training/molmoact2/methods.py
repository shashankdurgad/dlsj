"""The two MolmoAct2 post-training methods (both bf16-mixed, rank-64/alpha-16 LoRA, 5e-5 everywhere).

Both train the WHOLE action expert (578M params) and the WHOLE connector (57M) in full; LLM and ViT get LoRA.

A  "ai2"    Ai2's LoRA recipe (allenai/molmoact2 experiments/README.md, `--ft_vlm --ft_action_expert
            --ft_embedding=lm_head`): LoRA on the LLM and ViT block linears only; connector (image_pooling_2d +
            image_projector), action expert and the final LLM norm `ln_f` trained in full. `lm_head`/`wte` frozen.
B  "studio" Studio's default LoRA leaves on the LLM and ViT (also the ViT patch embedding); connector and action
            expert trained in full (Studio's own `unfreeze_action_expert()`); `ln_f` frozen.

Each class passes its adapter regex (`train.AI2_LORA_TARGETS` / `train.STUDIO_VLM_LORA_TARGETS`) as
lora_target_modules; Studio injects LoRA (freezing every base weight), then the class re-enables its full-tune
parts. That runs inside `_apply_model_modifications`, which Studio calls both when building the model and when
rebuilding it from a checkpoint, before `configure_optimizers` collects `requires_grad` parameters, so resume and
the optimizer's connector/action-expert LR groups see the same trainable set. Each class checks the result and
fails at startup if module names ever stop matching.

Both build on `policy_text.MolmoAct2Text128` (text padded to 128 tokens, trained in). Export/deploy needs no special
class: either checkpoint loads in `physicalai.policies.MolmoAct2` (the
hyperparameters carry `lora_target_modules`; full-tuned weights are ordinary state-dict entries) and `export()`
merges the adapters.
"""
import logging

from policy_text import MolmoAct2Text128  # 128-token static text, trained in (must match the XPS export)

log = logging.getLogger(__name__)


def group(name):
    if "action_expert" in name:
        return "action_expert"
    if "image_pooling_2d" in name or "image_projector" in name:
        return "connector"
    if "vision_backbone" in name:
        return "vit"
    return "llm"


def trainable_summary(model):
    """{group: {"lora": n, "full": n}} trainable parameter counts."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            kind = "lora" if "lora_" in name else "full"
            out.setdefault(group(name), {"lora": 0, "full": 0})[kind] += p.numel()
    return out


class _FullActionExpert(MolmoAct2Text128):
    LABEL = ""
    FULL_TUNE: tuple = ()                       # name fragments of base weights to train in full
    EXPECT_LORA: tuple = ()                     # groups that must carry adapters
    EXPECT_FULL: tuple = ()                     # groups that must have fully trained base weights
    NO_LORA = ("action_expert",)                # groups that must carry no adapters

    def _apply_model_modifications(self) -> None:
        super()._apply_model_modifications()
        if not self._require_config().use_lora:
            raise RuntimeError(f"{type(self).__name__} requires lora_enabled=True")
        model = self._require_model()
        model.unfreeze_action_expert()
        for name, p in model.named_parameters():
            if "lora_" not in name and any(part in name for part in self.FULL_TUNE):
                p.requires_grad = True
        s = trainable_summary(model)
        missing = [f"{g}:lora" for g in self.EXPECT_LORA if not s.get(g, {}).get("lora")]
        missing += [f"{g}:full" for g in self.EXPECT_FULL if not s.get(g, {}).get("full")]
        stray = [g for g in self.NO_LORA if s.get(g, {}).get("lora")]
        if missing or stray:
            raise RuntimeError(f"{self.LABEL} not applied as intended: missing {missing}, LoRA on {stray}: {s}")
        log.warning("%s trainable parameters: %s", self.LABEL, s)


class MolmoAct2Ai2Split(_FullActionExpert):
    LABEL = "Method A (Ai2 split)"
    FULL_TUNE = ("image_pooling_2d", "image_projector", "transformer.ln_f")
    EXPECT_LORA = ("llm", "vit")
    EXPECT_FULL = ("connector", "action_expert")
    NO_LORA = ("connector", "action_expert")


class MolmoAct2StudioLoraFullExpert(_FullActionExpert):
    LABEL = "Method B (Studio LLM+ViT LoRA, full connector + action expert)"
    FULL_TUNE = ("image_pooling_2d", "image_projector")
    EXPECT_LORA = ("llm", "vit")
    EXPECT_FULL = ("connector", "action_expert")
    NO_LORA = ("connector", "action_expert")
