import numpy as np
import pandas as pd
import pytest
import torch

from dlsj.warp import sampler, weights
from dlsj.warp.model import WarpRM, aggregate_velocity, two_hot, window_velocities

CFG = sampler.WarpConfig(n=16, stride_s=0.5, fps=30)


def test_window_fits_episode_and_labels_are_relative_displacement():
    rng = np.random.default_rng(0)
    for _ in range(300):
        idx, y = sampler.sample_window(CFG, 400, rng)
        assert len(idx) == CFG.n and idx.min() >= 0 and idx.max() <= 399
        np.testing.assert_allclose(y, (idx - idx[0]) / CFG.c_norm, atol=1e-6)


def test_sampler_produces_both_directions_and_speed_range():
    rng = np.random.default_rng(1)
    ends = np.array([sampler.sample_window(CFG, 2000, rng)[1][-1] for _ in range(2000)])
    assert (ends > 0).mean() == pytest.approx(0.5, abs=0.08)  # global flip p = 0.5 (plus reversals)
    assert np.abs(ends).max() <= 5 / 3 + 1e-6  # path length <= 5/3 L, i.e. |y| <= 5/3


def test_short_episode_still_yields_a_window():
    idx, _ = sampler.sample_window(CFG, 40, np.random.default_rng(2))
    assert idx.max() <= 39


def test_two_hot_round_trips_through_bin_centres():
    y = torch.tensor([-3.0, -0.37, 0.0, 1.0, 2.9, 5.0])
    t = two_hot(y, 30, -3.0, 3.0)
    assert torch.allclose(t.sum(-1), torch.ones(6))
    centres = torch.linspace(-3, 3, 30)
    assert torch.allclose((t * centres).sum(-1), y.clamp(-3, 3), atol=1e-5)


def test_model_shapes_and_loss_decreases_on_toy_progress():
    torch.manual_seed(0)
    m = WarpRM(feat_dim=8, n_frames=CFG.n, d_model=32, layers=2, heads=4, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    # toy "video": feature = a line in time, so progress is readable from features
    rng = np.random.default_rng(0)
    T = 300
    feats = torch.tensor(np.stack([np.linspace(0, 1, T)] * 8, 1), dtype=torch.float32)
    first = None
    for step in range(150):
        batch = [sampler.sample_window(CFG, T, rng) for _ in range(32)]
        f = torch.stack([feats[torch.as_tensor(i)] for i, _ in batch])
        y = torch.tensor(np.stack([l for _, l in batch]))
        loss = m.loss(f, y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.6 * first
    assert m.expected(f).shape == (32, CFG.n)


def test_aggregate_velocity_assigns_constant_pace():
    n, T = CFG.n, 400
    starts = np.arange(0, T - int((n - 1) * CFG.stride_frames))
    idx = np.stack([sampler.canonical_window(CFG, s) for s in starts])
    y_hat = (idx - idx[:, :1]) / CFG.c_norm  # a perfect model at canonical pace
    v = aggregate_velocity(T, starts, idx, window_velocities(y_hat, n))
    np.testing.assert_allclose(v[:-1], 1.0, atol=1e-6)


def test_weights_keep_hold_and_cover_every_phase():
    cfg = weights.WeightConfig(chunk=10, min_keep=0.3, phase_bins=5)
    fps, T, split = 30, 300, 150
    v = np.full(T, 1.5)
    v[140:200] = 0.0  # inspection hold: looks like no progress on camera
    v[20:60] = 0.2  # slow grasp alignment
    grip = np.r_[np.full(60, 40.0), np.full(T - 60, 3.0)]  # open (40) until the grasp at frame 60, then closed (3)
    fr = weights.episode_frame_table(0, v, split, grip, fps, cfg)
    df = weights.compute_weights(fr, {0: 1.0, 1: 1.0}, cfg)
    hold = df[(df.frame_index >= split - 15) & (df.frame_index < split + 15)]  # +-0.5 s
    assert (hold.weight >= 1.0).all()  # never dropped
    rep = weights.coverage_report(df)
    assert (rep["kept"] >= 0.3 - 1e-9).all()
    assert (df.loc[df.reason == "dropped", "weight"] == 0).all()


def test_stage_relative_pace_judges_each_stage_against_its_own_reference():
    cfg = weights.WeightConfig(chunk=1, min_keep=0.0)
    v = np.r_[np.full(100, 0.6), np.full(100, 1.4)]  # pick is slower on camera than place
    fr = weights.episode_frame_table(0, v, 100, None, 30, cfg)
    df = weights.compute_weights(fr, {0: 0.5, 1: 1.2}, cfg)
    assert (df.loc[df.stage == 0, "weight"] > 0).all()  # 0.6 / 0.5 = 1.2 > tau
    assert (df.loc[df.stage == 1, "weight"] > 0).all()


def _tiny_act():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = ACTConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(6,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=8, n_action_steps=8, dim_model=32, dim_feedforward=64, n_heads=4,
        n_encoder_layers=1, n_decoder_layers=1, latent_dim=8, n_vae_encoder_layers=1, device="cpu")
    return ACTPolicy(cfg)


def test_act_per_sample_loss_matches_lerobot_mean_loss():
    from dlsj.warp import train_policy

    train_policy.install()
    torch.manual_seed(0)
    pol = _tiny_act().train()
    B = 4
    batch = {"observation.state": torch.randn(B, 6), "observation.environment_state": torch.randn(B, 6),
             "action": torch.randn(B, 8, 6), "action_is_pad": torch.zeros(B, 8, dtype=torch.bool)}
    torch.manual_seed(1)
    mean_loss, _ = pol(batch)
    torch.manual_seed(1)
    per, _ = pol(batch, reduction="none")
    assert per.shape == (B,)
    assert torch.allclose(per.mean(), mean_loss, atol=1e-5)


def test_warp_weighter_looks_up_frame_weights(tmp_path):
    from dlsj.warp.train_policy import WarpWeights

    pd.DataFrame({"index": [0, 1, 2, 3], "weight": [0.0, 2.0, 1.0, 1.0]}).to_parquet(tmp_path / "w.parquet")
    ww = WarpWeights(str(tmp_path / "w.parquet"), torch.device("cpu"))
    w, stats = ww.compute_batch_weights({"index": torch.tensor([0, 1, 3])})
    assert torch.allclose(w, torch.tensor([0.0, 2.0, 1.0]) * 3 / 3.0)
    assert stats["kept_in_batch"] == pytest.approx(2 / 3)


def _fake_dataset(root, lengths, splits, success):
    import json

    (root / "meta" / "dlsj").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
             "gripper.pos"]
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "features": {"action": {"names": names}}}))
    rows, start = [], 0
    for ep, n in enumerate(lengths):
        for t in range(n):
            rows.append({"episode_index": ep, "frame_index": t, "index": start + t, "task_index": int(t >= splits[ep]),
                         "action": [0.0] * 5 + [80.0 if t > n // 3 else 0.0]})
        start += n
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    with open(root / "meta" / "dlsj" / "labels.jsonl", "w") as fh:
        for ep, ok in enumerate(success):
            fh.write(json.dumps({"episode_index": ep, "success": ok, "inspection_frame": splits[ep]}) + "\n")


def test_load_episodes_reads_split_success_gripper_and_offsets(tmp_path):
    from dlsj.warp.data import load_episodes

    _fake_dataset(tmp_path, [90, 120], [40, 70], [True, False])
    eps, fps = load_episodes(tmp_path)
    assert fps == 30 and [e.length for e in eps] == [90, 120] and [e.start for e in eps] == [0, 90]
    assert [e.split for e in eps] == [40, 70] and [e.success for e in eps] == [True, False]
    assert eps[0].gripper[-1] == 80.0


def test_episode_velocity_handles_episodes_shorter_than_a_window():
    from dlsj.warp.score import episode_velocity

    cfg = sampler.WarpConfig(n=16, stride_s=0.5, fps=30)
    m = WarpRM(feat_dim=4, n_frames=16, d_model=16, layers=1, heads=2, dropout=0.0).eval()
    for T in (60, 400):
        v = episode_velocity(m, cfg, np.random.default_rng(0).normal(size=(T, 4)).astype(np.float32), "cpu")
        assert v.shape == (T,) and np.isfinite(v).all()


def test_chunk_ending_in_place_is_judged_by_place_pace():
    cfg = weights.WeightConfig(chunk=50, min_keep=0.0)
    v = np.r_[np.full(100, 0.5), np.full(100, 1.0)]
    fr = weights.episode_frame_table(0, v, 100, None, 30, cfg)
    df = weights.compute_weights(fr, {0: 0.4, 1: 2.0}, cfg)
    late_pick = df[(df.frame_index >= 55) & (df.frame_index < 70)]  # end in place (1.0/2.0 = 0.5), before the hold window
    assert (late_pick.stage_end == 1).all() and (late_pick.weight == 0).all()
    early_pick = df[df.frame_index < 40]  # chunks end in pick: 0.5/0.4 = 1.25 > tau
    assert (early_pick.weight > 0).all()


def test_train_rm_and_score_end_to_end_on_fake_data(tmp_path, monkeypatch):
    import json
    import sys

    from dlsj.warp import score, train_rm

    root = tmp_path / "ds"
    n_eps = 24
    lengths = [150 + 10 * (i % 6) for i in range(n_eps)]
    _fake_dataset(root, lengths, [L // 2 for L in lengths], [True] * n_eps)
    for cam in ("top", "wrist"):
        (tmp_path / "feats" / cam).mkdir(parents=True)
        for ep, L in enumerate(lengths):
            t = np.linspace(0, 1, L)[:, None]
            np.save(tmp_path / "feats" / cam / f"episode_{ep:06d}.npy", (t * np.ones((1, 8))).astype(np.float16))
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--feats", str(tmp_path / "feats"), "--out",
                                      str(tmp_path / "rm"), "--steps", "60", "--bs", "16", "--eval-every", "30",
                                      "--d-model", "32", "--layers", "1", "--device", "cpu", "--ref-quantile", "1.0"])
    train_rm.main()
    summ = json.loads((tmp_path / "rm" / "train_summary.json").read_text())
    assert summ["best_step"] in (30, 60)
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--feats", str(tmp_path / "feats"), "--rm",
                                      str(tmp_path / "rm" / "warp_rm.pt"), "--out", str(tmp_path / "w"),
                                      "--chunk", "30", "--device", "cpu"])
    score.main()
    w = pd.read_parquet(tmp_path / "w" / "warp_weights.parquet")
    assert len(w) == sum(lengths) and set(w.reason) <= {"pace", "protected", "coverage", "dropped"}
    assert w["index"].is_monotonic_increasing


def test_only_the_successful_grasp_and_the_release_are_protected():
    cfg = weights.WeightConfig()
    fps = 30
    #   open, fumble close @30, re-open @60, real close @100, split @150, release @200
    grip = np.r_[np.full(30, 40.0), np.full(30, 3.0), np.full(40, 40.0), np.full(100, 3.0), np.full(60, 40.0)]
    moments = weights.key_grip_moments(grip, 150, fps, cfg)
    assert len(moments) == 2
    grasp, release = moments
    assert 85 <= grasp <= 100 and 185 <= release <= 200
    fr = weights.episode_frame_table(0, np.zeros(len(grip)), 150, grip, fps, cfg)
    assert not fr.loc[(fr.frame_index > 25) & (fr.frame_index < 70), "protected"].any()  # fumble not protected
