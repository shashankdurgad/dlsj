"""Isaac-side harness of the visual fit: TwinEnv posed at the recorded joint state of every evaluation sample.

    from sim.twin.visuals.fit.harness import Harness      # inside sim/twin/run.sh (launches Isaac through TwinEnv)
    h = Harness(look=None)
    out = h.render_set(ids, masks=True)                   # {cam: (N, H, W, 3) raw renders, "mask_<cam>": (N, H, W)}

Per sample: the recorded joint state (TwinEnv.reset(joint_deg=...)), the plates and bricks where the real frame-0 top
image shows them (evalset.json; detected colours bound through twin_scene.set_brick_color), the target brick on the
table before the grasp and between the fingertips afterwards (visual proxy placed at the fingertip frame). Renders are
RAW (TwinEnvCfg.color_correct=False); apply twin_scene.color_correct for the camera response.
Region masks come from the sim's own geometry: robot / bricks / plates by hiding them and differencing the render,
the table from the camera ray hitting the real table rectangle. Bits: 1 table, 2 plates, 4 bricks, 8 robot.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "sim/twin/out/visual_fit"
CAMS = ("top", "wrist", "inspect")
HELD_MOMENTS = ("grasp", "inspect", "place")
TABLE_RECT = (-0.06, 0.90, -0.65, 0.65)  # x0, x1, y0, y1 of real table wood the masks count as "table" (env frame)


class Harness:
    """TwinEnv + evaluation samples: pose, render, region masks (see the module doc)."""

    def __init__(self, look=None, width: int = 640, height: int = 480, passes: int = 2, cfg_kw: dict | None = None):
        from sim.twin.env import TwinEnv, TwinEnvCfg

        self.env = TwinEnv(TwinEnvCfg(width=width, height=height, cameras=CAMS, meas_noise=False,
                                      max_relative_target=1000.0, color_correct=False, look=look,
                                      render_passes=passes, **(cfg_kw or {})))
        import torch

        self.torch = torch
        meta = json.loads((OUT / "evalset.json").read_text())
        self.samples = meta["samples"]
        self.W, self.H = width, height
        self.held = None  # (pos (3,), quat xyzw (4,)) of the held target brick's proxy
        self._targets: dict[int, dict] = {}
        env = self.env
        orig = env.sync

        def sync():
            n = orig()
            if self.held is not None:
                from sim.twin.visuals.usd_sync import _mat, _parent_world, set_local

                prim = env.stage.GetPrimAtPath("/World/envs/env_0/TwinVis/Brick")
                if prim:
                    set_local(prim, _mat(*self.held) * _parent_world(prim).GetInverse())
            return n

        env.sync = sync

    # ------------------------------------------------------------------ posing
    def _tip(self):
        R, env = self.env.residual, self.env
        tip = (R.tip_w() - env.scene.env_origins)[0].detach().cpu().numpy()
        q = R.robot.data.body_quat_w[0, R.grip_body].detach().cpu().numpy()
        return tip, q

    def _target(self, s) -> dict:
        ep = s["ep"]
        if ep in self._targets:
            return self._targets[ep]
        env = self.env
        env.reset(seed=ep, joint_deg=s["grasp_state"], brick_pose=(0.8, 0.5, 0.0), render=False)
        tip, _ = self._tip()
        best, bd = None, 0.06
        for k, b in enumerate(s["bricks"]):
            d = float(np.hypot(b["x"] - tip[0], b["y"] - tip[1]))
            if d < bd:
                best, bd = k, d
        t = dict(index=best, xy=(float(tip[0]), float(tip[1])), yaw=0.0, color=None)
        if best is not None:
            b = s["bricks"][best]
            t.update(xy=(b["x"], b["y"]), yaw=b["yaw"], color=b["color"])
        self._targets[ep] = t
        return t

    def pose(self, s):
        """Put the twin in evaluation sample `s`: joints, plates, bricks (target on the table or in the gripper)."""
        env, torch = self.env, self.torch
        from sim.twin.visuals.twin_scene import set_brick_color

        R = env.residual
        t = self._target(s)
        held = s["moment"] in HELD_MOMENTS
        bp = (0.8, 0.5, 0.0) if held else (*t["xy"], t["yaw"])
        env.reset(seed=s["ep"], joint_deg=s["state"], brick_pose=bp, render=False)
        if t["color"]:
            set_brick_color(0, "Brick", t["color"])
        o = env.scene.env_origins[:1]
        dev = env.device

        def put(obj, x, y, z=None, yaw=0.0):
            d = obj.data
            pose = d.root_pose_w[:1].clone() if hasattr(d, "root_pose_w") else d.root_state_w[:1, :7].clone()
            pose[:, 0], pose[:, 1] = x + o[0, 0], y + o[0, 1]
            if z is not None:
                pose[:, 2] = z + o[0, 2]
            pose[:, 3:] = torch.tensor([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], device=dev)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

        for k, (name, obj) in enumerate(zip(R.bowl_names, R.bowl_objs, strict=True)):
            if name in s["plates"]:
                put(obj, s["plates"][name][0], s["plates"][name][1])
            else:
                put(obj, 3.0, -0.55 + 0.3 * k, -1.5)
        rest = [b for k, b in enumerate(s["bricks"]) if k != t["index"]]
        free = list(range(len(R.distractors)))
        sizes = [env.layout.distractors[i].size for i in free]
        for b in rest:
            want_long = b["kind"] == "2x4"
            pick = next((i for i in free if (max(sizes[i][:2]) > 0.05) == want_long), None)
            if pick is None:
                continue
            free.remove(pick)
            z = float(R.table_z) + sizes[pick][2] / 2 + 0.0005 + (0.004 if b["on_plate"] else 0.0)
            put(R.distractors[pick], b["x"], b["y"], z, b["yaw"])
            set_brick_color(0, f"Distractor{pick}", b["color"])
        for i in free:
            put(R.distractors[i], 3.0, -0.4 + 0.1 * i, -1.5)
        env.settle(2)
        self.held = self._tip() if held else None

    # ------------------------------------------------------------------ rendering
    def _vis(self, paths, on: bool):
        from pxr import UsdGeom

        for p in paths:
            prim = self.env.stage.GetPrimAtPath(p)
            if prim:
                im = UsdGeom.Imageable(prim)
                im.MakeVisible() if on else im.MakeInvisible()

    def _groups(self):
        st = self.env.stage
        tv = st.GetPrimAtPath("/World/envs/env_0/TwinVis")
        kids = [str(c.GetPath()) for c in tv.GetChildren()] if tv else []
        return {8: ["/World/envs/env_0/RobotVisual"], 4: [k for k in kids if "/Bowl_" not in k],
                2: [k for k in kids if "/Bowl_" in k]}

    def cam_rays_table(self, cam: str):
        """Bool (H, W): the pixel's ray hits the table plane inside TABLE_RECT."""
        from pxr import Usd, UsdGeom

        from sim.twin.visuals.twin_scene import wrist_cam_rel

        rel = {"top": "TopCam", "inspect": "InspectCam", "wrist": wrist_cam_rel()}[cam]
        prim = self.env.stage.GetPrimAtPath(f"/World/envs/env_0/{rel}")
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        c = UsdGeom.Camera(prim)
        fx = c.GetFocalLengthAttr().Get() / c.GetHorizontalApertureAttr().Get() * self.W
        v, u = np.mgrid[:self.H, :self.W]
        d = np.stack([(u + 0.5 - self.W / 2) / fx, -(v + 0.5 - self.H / 2) / fx, -np.ones_like(u, float)], -1)
        dw = d @ M[:3, :3]
        pos = M[3, :3] - self.env.scene.env_origins[0].detach().cpu().numpy()
        z = float(self.env.residual.table_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - pos[2]) / dw[..., 2]
        P = pos + t[..., None] * dw
        x0, x1, y0, y1 = TABLE_RECT
        return (t > 0) & (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)

    def render(self):
        """One raw render of all cameras: {cam: (H, W, 3) uint8}."""
        return {c: v.copy() for c, v in self.env.render().items()}

    def masks(self, base: dict) -> dict:
        """Region bit-masks per camera for the current pose (`base` = its render): hide a group, difference."""
        import cv2

        out = {c: np.zeros((self.H, self.W), np.uint8) for c in base}
        for bit, paths in self._groups().items():
            self._vis(paths, False)
            img = self.render()
            self._vis(paths, True)
            for c in base:
                d = np.abs(img[c].astype(np.int16) - base[c].astype(np.int16)).max(-1) > 22
                d = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                out[c][d > 0] |= bit
        for c in base:
            occ = cv2.dilate((out[c] > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            out[c][self.cam_rays_table(c) & ~occ] |= 1
        self.render()  # everything visible again in the renderer's history
        return out

    def render_set(self, ids=None, masks: bool = False, progress: bool = True) -> dict:
        """Render samples `ids` (default all): {cam: (N, H, W, 3)} (+ "mask_<cam>": (N, H, W) with `masks`)."""
        ids = list(range(len(self.samples))) if ids is None else list(ids)
        res = {c: [] for c in CAMS}
        mk = {c: [] for c in CAMS}
        for n, i in enumerate(ids):
            if getattr(self, "verbose", False):
                s = self.samples[i]
                print(f"[harness] sample {i} ep {s['ep']} {s['moment']}", flush=True)
            self.pose(self.samples[i])
            self.render()  # first frame after the teleport (temporal history of the renderer)
            img = self.render()
            for c in CAMS:
                res[c].append(img[c])
            if masks:
                m = self.masks(img)
                for c in CAMS:
                    mk[c].append(m[c])
            if progress and n % 25 == 0:
                print(f"[harness] rendered {n + 1}/{len(ids)}", flush=True)
        out = {c: np.stack(v) for c, v in res.items()}
        if masks:
            out.update({"mask_" + c: np.stack(v) for c, v in mk.items()})
        return out

    def close(self):
        """Close the env (and the Kit app it launched)."""
        self.env.close()
