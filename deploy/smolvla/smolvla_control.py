"""MANUAL / TESTING CONTROLLER. The loop we actually run is `auto_control.py` (wraps this file and adds the
automatic spin-and-inspect + bowl decision). Use this one only to test the VLA by hand (M + 1/2/3); keep its classes
and main() stable because auto_control.py patches `make_controller_class` and calls `main()`.

SmolVLA (3-stage, wrist + top) on the SO-101 via Desmond's console (`dlsj.deploy.pi05_control`), with RTC.

Reuses that controller (web console :8010, speed limits, Q = hold, ramp home on exit) and changes:
* prompts: the 3-stage strings the model was trained on. Space = "pick up the brick"; M (once grasped) = "bring the
  brick to the inspection camera"; M again = "place the brick in the <bowl> bowl" (choose the bowl with 1/2/3 first);
* cameras: the policy reads only `wrist` and `top` (SmolVLA: full frame, no crop; MolmoAct2 fine-tune: `--crop
  top=0,0,560,480`, as its training set);
* `{brick}` in `--pick-prompt` is filled from `--brick` (e.g. "pick up the {brick} brick" -> "pick up the red brick");
  the console page can change the pick/place prompt live;
* `--training-tasks FILE` (json with "tasks", e.g. deploy/molmoact2/finetune_cal2_456.json): every prompt the console
  can send (pick, bring, place for each bowl, and live edits) must be one of the training task strings, and the
  training top crop must be passed; `--check-only` validates and exits without touching the arm;
* `--joint-offset joint=deg,...`: current calibration minus the training-time one (as in act_rollout); the policy
  sees state - offset, the arm gets action + offset;
* RTC from `deploy/pi05/rtc_control.py` when the export has RTC inputs (`deploy/smolvla/export.py --rtc`).
* `--residual off|shadow|on --residual-weights <residual_actor.npz>`: the pick-only grasp residual (`residual.py`),
  hooked into the goal of every policy tick while the pick prompt is active; off (default) = unchanged.

    cd ~/inference/dlsj-smolvla && PYTHONPATH=$PWD ~/desmond/venvs/pi05rt/bin/python \
        ~/inference/deploy/smolvla/smolvla_control.py --export ~/inference/exports/smolvla_3stage_rtc_int8 \
        --device GPU --rtc-horizon 15 --rtc-guidance 3 --joint-offset recal_2026-11-14 --max-step 4 \
        --cameras wrist=/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0 \
        --realsense top=346422061366,inspect=243722062690
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05"))
sys.path.insert(0, str(Path("~/inference/deploy/pi05").expanduser()))
import rtc_control as R  # noqa: E402

log = logging.getLogger("smolvla_control")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PICK = "pick up the brick"
BRING = "bring the brick to the inspection camera"
PLACE = "place the brick in the {colour} bowl"
BRICK = ""
PICK_TEMPLATE = PICK  # --pick-prompt before {brick} is filled: the brick buttons refill it
BRICK_COLOURS = ["blue", "brown", "red", "white", "yellow"]  # training colours (contract brick_colours when given)
STAGES = ("pick", "bring", "place")
TRAINING_TASKS: set[str] | None = None  # set by --training-tasks: prompts outside it are refused
HOME_SPEED = 30.0  # deg/s for the Home button


CHOOSER = """
<div class="chooser"><div class="lbl">Brick</div><div class="plates" id="bricks"></div>
<div class="lbl">Stage</div><div class="btns" id="stages"></div></div>
<style>.chooser .lbl{font-size:12px;opacity:.7;margin:8px 0 4px}#bricks button,#stages button{cursor:pointer}
#stages button.sel{outline:3px solid #4a9;font-weight:600}#bricks button.sel{outline:3px solid #4a9}
#bricks button{min-width:58px;min-height:34px;border-radius:8px;border:1px solid #888;font-size:13px}</style>
<script>(()=>{const BC={blue:'#3d7be0',brown:'#8b5a2b',red:'#d33b3b',white:'#f4f4f4',yellow:'#f2c230'};
const NM={pick:'Pick',bring:'Bring to camera',place:'Place'};
async function chooser(){try{const r=await fetch('/ctl/state').then(x=>x.json());
 const b=document.getElementById('bricks'),s=document.getElementById('stages');
 if(b&&!b.children.length)(r.brick_colours||[]).forEach(c=>{const e=document.createElement('button');e.textContent=c;
  e.style.background=BC[c]||c;e.style.color=(c==='white'||c==='yellow')?'#222':'#fff';e.onclick=()=>cmd('brick_'+c);b.appendChild(e)});
 if(s&&!s.children.length)(r.stages||[]).forEach(st=>{const e=document.createElement('button');e.className='k';e.dataset.st=st;
  e.textContent=NM[st]||st;e.onclick=()=>cmd('stage_'+st);s.appendChild(e)});
 if(b)[...b.children].forEach(e=>e.classList.toggle('sel',e.textContent===r.brick));
 const cur=(r.phase==='running'||r.phase==='readying')?r.stage:r.start_stage;
 if(s)[...s.children].forEach(e=>e.classList.toggle('sel',e.dataset.st===cur));
 const p=document.getElementById('prompt');if(p&&r.prompt)p.textContent=r.prompt;}catch(e){}}
setInterval(chooser,300);chooser();})();</script>"""


def add_home_button(pc) -> None:
    """Home button + H key on the console page, and 'home' accepted by its HTTP handler."""
    btn = """<button class="k" onclick="cmd('mark')">Mark inspection <kbd>M</kbd></button>"""
    key = "else if(k==='m')cmd('mark');"
    page = getattr(pc, "PAGE", "")
    if btn in page and key in page:
        page = page.replace(btn, btn + """<button class="k" onclick="cmd('home')">Home <kbd>H</kbd></button>""") \
                   .replace(key, key + "else if(k==='h')cmd('home');")
        pc.PAGE = page.replace('<div class="plates" id="plates"></div>', '<div class="plates" id="plates"></div>' + CHOOSER)
    else:
        log.warning("console page layout not recognised: no Home button (POST /ctl/home still works)")
    make_handler = pc.make_handler

    def handler_with_home(streams, ctl):
        base = make_handler(streams, ctl)

        class Handler(base):
            def do_POST(self):
                name = self.path.removeprefix("/ctl/")
                if name in ("home", "h") or name.startswith(("brick_", "stage_", "residual_")):
                    ctl.command(name)
                    return self._send(200, b"{}", "application/json")
                return super().do_POST()

        return Handler

    pc.make_handler = handler_with_home


def fill(template: str, brick: str = "", colour: str = "") -> str:
    return template.replace("{brick}", brick).replace("{colour}", colour)


def prompts_outside(tasks: set[str], pick: str, bring: str, place: str, bowls, brick: str) -> list[str]:
    """Every prompt the console can send with these settings that is not a training task string."""
    want = [fill(pick, brick=brick)] + ([bring] if bring else []) + [fill(place, colour=b) for b in bowls]
    return [w for w in want if w not in tasks]


def _arg(rest: list[str], name: str) -> str | None:
    """Value of `--name v` / `--name=v` in the console's own arguments."""
    for i, a in enumerate(rest):
        if a == name and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def parse_offset(spec: str) -> np.ndarray:
    """joint=deg pairs, or a name from configs/joint_offsets.json (e.g. recal_2026-11-14), like act_rollout."""
    off = np.zeros(6)
    if spec and "=" not in spec:
        import json

        from dlsj import arms

        table = json.loads((Path(arms.__file__).resolve().parents[1] / "configs" / "joint_offsets.json").read_text())
        if spec not in table or spec.startswith("_"):
            raise SystemExit(f"--joint-offset: no entry {spec!r} in configs/joint_offsets.json")
        spec = ",".join(f"{k}={v}" for k, v in table[spec]["offset_deg"].items())
    for kv in filter(None, (s.strip() for s in spec.split(","))):
        k, v = kv.split("=")
        if k not in JOINTS:
            raise SystemExit(f"--joint-offset: unknown joint {k} (joints: {JOINTS})")
        off[JOINTS.index(k)] = float(v)
    return off


def make_policy_class(base, offset: np.ndarray):
    class SmolPolicy(base):
        """Pi05Policy-compatible wrapper: only the cameras the export asks for; calibration offset applied."""

        def __call__(self, state, images, task):
            s = np.asarray(state, np.float32) - offset.astype(np.float32)
            a = super().__call__(s, images, task)
            return np.asarray(a) + offset

    return SmolPolicy


def make_controller_class(base):
    class SmolController(base):
        def prompt(self) -> str:
            # newer consoles keep editable prompts (pick_prompt/place_prompt, set by the page); older ones don't
            if self.stage == "pick":
                if "{brick}" in PICK_TEMPLATE:  # the brick-colour buttons choose the colour
                    return fill(PICK_TEMPLATE, brick=getattr(self, "brick", "") or BRICK)
                return getattr(self, "pick_prompt", None) or PICK
            if self.stage == "bring":
                return BRING
            return fill(getattr(self, "place_prompt", None) or PLACE, colour=self.bowl or "yellow")

        def set_prompts(self, pick, place) -> None:
            if TRAINING_TASKS:
                bad = []
                if isinstance(pick, str) and pick.strip() and pick.strip() not in TRAINING_TASKS:
                    bad.append(pick.strip())
                if isinstance(place, str) and place.strip() and any(
                        fill(place.strip(), colour=b) not in TRAINING_TASKS for b in self.bowls):
                    bad.append(place.strip())
                if bad:
                    self.message = f"Not a training prompt, ignored: {bad}"
                    log.warning("refused prompt edit (not in --training-tasks): %s", bad)
                    return
            if hasattr(super(), "set_prompts"):
                super().set_prompts(pick, place)

        def _handle(self, c: str) -> None:
            if c in ("home", "h"):  # Home button / H: stop the policy and ease back to the saved home pose
                home = self.arm.arms.load_home(self.arm.id) if hasattr(self.arm, "arms") else None
                if not home:
                    self.message = "No saved home pose for this follower"
                    return
                self._home = np.array([float(home[j]) for j in self.arm.joints], dtype=float)
                self._replan_now()
                self.stage = "pick"
                self.cmd = np.asarray(self.arm.read(), dtype=float)
                self.phase, self.message = "homing", "Going home"
                return
            if c.startswith("brick_"):
                colour = c.removeprefix("brick_")
                if "{brick}" not in PICK_TEMPLATE or colour not in BRICK_COLOURS:
                    self.message = f"Brick colour {colour!r} not available"
                    return
                self.brick = colour
                if self.phase == "running" and self.stage == "pick":
                    self._replan_now()
                self.message = f"Brick: {colour}"
                return
            if c.startswith("stage_") and c.removeprefix("stage_") in STAGES:
                stage = c.removeprefix("stage_")
                if stage == "bring" and not BRING:
                    self.message = "This model has no bring stage"
                    return
                if stage == "place" and self.bowl is None:
                    self.message = "Choose a plate (1-3) first"
                    return
                self.start_stage = stage
                if self.phase in ("running", "readying"):
                    self.stage = stage
                    self._replan_now()
                    self.message = f"Stage: {self.prompt()}"
                else:
                    self.message = f"Space starts at: {stage}"
                return
            if c in ("start", "space") and self.phase not in ("running", "readying"):
                super()._handle(c)
                self.stage = getattr(self, "start_stage", "pick")
                return
            # The console only sends space / m / 1-3 / +- / q / home, so M advances the stage:
            # pick --M--> bring (brick grasped) --M--> place (needs a bowl chosen with 1/2/3).
            if c in ("g", "grasp", "bring", "mark", "m") and self.phase == "running" and self.stage == "pick" and BRING:
                self.stage = "bring"
                self._replan_now()
                self.message = "Stage: bring the brick to the inspection camera (M again to place)"
                return
            if c in ("mark", "m") and self.phase == "running" and self.stage in (("bring",) if BRING else ("pick",)):
                if self.bowl is None:
                    self.message = "Choose a plate (1-3), then M to place"
                    return
                self.stage = "place"
                self._replan_now()
                return
            super()._handle(c)

        def tick(self) -> None:
            super().tick()  # handles the keys (Q stops homing too) and the readying/running phases
            with self.lock:
                if self.phase == "homing":
                    base = self.cmd if self.cmd is not None else np.asarray(self.arm.read(), dtype=float)
                    step = HOME_SPEED / max(1, self.fps)
                    measured = self._step(base + np.clip(self._home - base, -step, step))
                    if np.abs(measured - self._home).max() < 3.0:
                        self._replan_now()
                        self.phase, self.message = "idle", "Home. Press Space to start"

        def snapshot(self) -> dict:
            data = super().snapshot()
            data.update(brick=getattr(self, "brick", "") or BRICK,
                        brick_colours=BRICK_COLOURS if "{brick}" in PICK_TEMPLATE else [],
                        stage=self.stage, start_stage=getattr(self, "start_stage", "pick"),
                        stages=[st for st in STAGES if st != "bring" or BRING], prompt=self.prompt())
            return data

        def _obs(self):
            state, imgs = super()._obs()
            return state, {k: v for k, v in imgs.items() if k in ("wrist", "top")}

    return SmolController


def _residual_args(ap) -> None:
    """The pick-only grasp residual's flags (deploy/smolvla/residual.py). Kept here so --residual off never imports
    it: off = this controller exactly as before."""
    ap.add_argument("--residual", choices=("off", "shadow", "on"), default="off",
                    help="pick-only grasp residual: off (default, controller unchanged), shadow (run + log, not "
                         "applied), on (applied while the pick prompt is active)")
    ap.add_argument("--residual-weights", default="zero",
                    help="residual_actor.npz (or its deploy/ dir) from export_residual.py; 'zero' = identity")
    ap.add_argument("--residual-device", default="numpy", help="numpy (default), or an OpenVINO device for the .onnx")
    ap.add_argument("--residual-detector", default="~/inference/models/brick-detector-v1",
                    help="wrist brick detector (OpenVINO model dir), run on --residual-detector-device")
    ap.add_argument("--residual-detector-device", default="NPU")
    ap.add_argument("--residual-max-joint-delta", type=float, default=20.0,
                    help="reject a corrected goal that moves any body joint more than this (deg) from the plan")
    ap.add_argument("--residual-top-seed", action="store_true",
                    help="seed the wrist tracker and the top-target obs from the loop's top-camera target (off by "
                         "default: the twin's top camera and the W9 FK disagree by ~5-7 cm at table level, see residual.py)")


def _residual_config(ours, rest=(), offset=None):
    """None for --residual off, else (residual module, ResidualConfig) with the weights validated and checked against
    this controller (before the arm is touched): the slew it was trained behind (--max-step / --max-step-gripper;
    "on" refuses a different one, "shadow" warns) and the calibration frame (no --joint-offset)."""
    if ours.residual == "off":
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import residual as RES

    cfg = RES.config_from_args(ours)
    log.info(RES.describe(cfg))  # loads + validates the export's meta
    if offset is not None and np.any(offset):
        raise SystemExit("--residual needs the arm in the calibration frame it was trained in: drop --joint-offset")
    pol = RES.load_policy(cfg.weights, cfg.device)
    default = 8.0  # pi05_control.DEFAULT_MAX_STEP (both flags)
    ms, msg = (float(_arg(list(rest), k) or default) for k in ("--max-step", "--max-step-gripper"))
    bad = RES.check_controller(pol.meta, ms, msg)
    for b in bad:
        log.warning("residual: %s", b)
    if bad and cfg.mode == "on":
        raise SystemExit("--residual on: " + "; ".join(bad) + " (run the controller it was trained behind, or shadow)")
    cfg.slew = f"max_step {ms:g},{msg:g}: " + ("; ".join(bad) if bad else "matches the training controller")
    return RES, cfg


def main() -> None:
    global PICK, BRING, PLACE, BRICK, TRAINING_TASKS
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--joint-offset", default="")
    ap.add_argument("--rtc-horizon", type=int, default=15)
    ap.add_argument("--rtc-guidance", type=float, default=3.0)
    ap.add_argument("--pick-prompt", default=PICK)
    ap.add_argument("--bring-prompt", default=BRING, help="'' = no bring stage (M goes pick -> place)")
    ap.add_argument("--place-prompt", default=PLACE, help="with {colour}")
    ap.add_argument("--brick", default="", help="brick colour for {brick} in --pick-prompt (e.g. red)")
    ap.add_argument("--training-tasks", default="", help="json with 'tasks' (and 'top_crop'): refuse other prompts")
    ap.add_argument("--check-only", action="store_true", help="validate prompts/crop/export and exit (no arm)")
    _residual_args(ap)
    ours, rest = ap.parse_known_args()
    global PICK_TEMPLATE, BRICK_COLOURS
    PICK, BRING, PLACE, BRICK = ours.pick_prompt, ours.bring_prompt, ours.place_prompt, ours.brick
    PICK_TEMPLATE = PICK
    if "{brick}" in PICK and not BRICK:
        raise SystemExit("--pick-prompt has {brick}: pass --brick <colour>")
    PICK = fill(PICK, brick=BRICK)
    offset = parse_offset(ours.joint_offset)
    bowls = (_arg(rest, "--bowls") or "yellow,purple,orange").split(",")
    contract = {}
    if ours.training_tasks:
        import json as _json

        contract = _json.loads(Path(ours.training_tasks).expanduser().read_text())
        TRAINING_TASKS = set(contract["tasks"])
        if contract.get("brick_colours"):
            BRICK_COLOURS = [c for c in contract["brick_colours"] if fill(PICK_TEMPLATE, brick=c) in TRAINING_TASKS]
        bad = prompts_outside(TRAINING_TASKS, PICK, BRING, PLACE, bowls, BRICK)
        if bad:
            raise SystemExit(f"prompts not in the training tasks ({ours.training_tasks}): {bad}\n"
                             f"training tasks: {sorted(TRAINING_TASKS)}")
        want_crop = contract.get("top_crop")
        crop = _arg(rest, "--crop") or ""
        if want_crop and f"top={want_crop}" not in crop:
            raise SystemExit(f"the model was trained with the top camera cropped to {want_crop}: pass "
                             f"--crop top={want_crop} (got {crop!r})")
    export = Path(_arg(rest, "--export") or "").expanduser()
    try:
        import json as _json

        export_policy = _json.loads((export / "manifest.json").read_text())["policy"]["name"]
    except Exception:  # noqa: BLE001
        export_policy = ""
    residual = _residual_config(ours, rest, offset)
    if ours.check_only:
        print(f"export {export} ({export_policy or 'no manifest'}); prompts: pick {PICK!r} / bring {BRING!r} / "
              f"place {[fill(PLACE, colour=b) for b in bowls]}; crop {_arg(rest, '--crop')!r}; "
              f"training tasks {'checked' if TRAINING_TASKS else 'not checked'}")
        if residual is not None:
            RES, cfg = residual
            print(RES.describe(cfg), "| gate on the pick prompt:", RES.is_pick_prompt(PICK), "|", cfg.slew)
            print(RES.probe(cfg))  # wrist detector on its device + the vendored twin files; never the arm
        if not (export / "manifest.json").exists():
            raise SystemExit(f"no export at {export}")
        return

    from dlsj.deploy import pi05_control as pc
    from dlsj.deploy.pi05_runtime import Pi05Policy

    if "--crop" in " ".join(rest) and export_policy != "molmoact2":
        raise SystemExit("SmolVLA was trained on the full top frame: drop --crop")
    pc.PICK = PICK  # used by warmup() and the console's default prompt
    pc.PLACE = PLACE
    base_policy = make_policy_class(Pi05Policy, offset)
    use_rtc = True
    try:
        policy_cls = R.make_rtc_policy_class(base_policy)
    except Exception:  # noqa: BLE001
        use_rtc = False
    ctl_cls = make_controller_class(R.make_rtc_controller_class(pc.Controller))
    if residual is not None:  # outermost: smolvla / auto / loop all get it; --residual off leaves ctl_cls untouched
        RES, cfg = residual
        ctl_cls = RES.make_residual_controller_class(ctl_cls, cfg)

    def make_policy(args):
        # MolmoAct2: its own wrapper (RTC inputs are added after the runtime preprocessors, which drop unknown keys;
        # plain exports fall back inside the wrapper without a second 5B-model load)
        try:
            import json as _json

            name = _json.loads((Path(args.export).expanduser() / "manifest.json").read_text())["policy"]["name"]
        except Exception:  # noqa: BLE001
            name = ""
        if name == "molmoact2":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "molmoact2"))
            sys.path.insert(0, str(Path("~/inference/deploy/molmoact2").expanduser()))
            from rtc_patch import make_molmo_rtc_policy_class

            if offset.any():
                log.warning("--joint-offset ignored for MolmoAct2 (its export maps the current calibration itself)")
            pol = make_molmo_rtc_policy_class(Pi05Policy)(args.export, args.device, cache_dir=args.cache_dir,
                                                         horizon=ours.rtc_horizon, guidance=ours.rtc_guidance)
            log.info("MolmoAct2 policy: %s", "RTC" if pol.rtc_enabled else "plain chunking (export has no RTC inputs)")
            return pol
        # RTC if the loaded model takes the RTC inputs (checked on the model's input features, not the manifest
        # text: some exports don't spell the RTC inputs out in manifest.json)
        try:
            return policy_cls(args.export, args.device, cache_dir=args.cache_dir, horizon=ours.rtc_horizon,
                              guidance=ours.rtc_guidance)
        except SystemExit as e:
            log.warning("%s -> plain chunking", e)
            return base_policy(args.export, args.device, cache_dir=args.cache_dir)

    def controller(*a, **kw):
        c = ctl_cls(*a, horizon=ours.rtc_horizon, **kw)
        if not getattr(c.policy, "rtc_enabled", hasattr(c.policy, "rtc_next")):
            c.blend = "crossfade"
        return c

    pc.make_policy = make_policy
    pc.Controller = controller
    add_home_button(pc)
    sys.argv = [sys.argv[0], *rest]
    logging.basicConfig(level=logging.INFO)
    log.info("prompts: %r / %r / %r", PICK, BRING, PLACE)
    log.info("SmolVLA 3-stage: offset %s, RTC h=%d g=%.1f", dict(zip(JOINTS, offset)), ours.rtc_horizon,
             ours.rtc_guidance)
    pc.main()


if __name__ == "__main__":
    main()
