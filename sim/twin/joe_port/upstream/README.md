# upstream/ — files copied from Joe's Isaac Sim setup (read-only survey, 15 Nov 2026)

These files are Joe's work, not ours. Source: `/home/joe-spark-2/pi05-isaac-so101/` on Joe's GB10 (joe-spark-2),
copied with the same relative paths. Contents are unmodified; every `*.py` is stored as `*.py.txt` so the
twin lint (`sim/twin/lint.sh`) and Python tooling skip it. Last change in his tree: 14 Nov 2026 14:40. Kept for study and as
evidence for `../README.md`; nothing here is imported by the twin. His scripts are fragments that run inside a
persistent Isaac Sim 6.2 session (they assume globals such as `stage`, `robot`, `indices`, `limits`, `reference`),
so they do not run stand-alone.

What was copied
* top level: all scripts (`setup_scene.py`, `configure_reference.py`, `fix_wrist*.py`, `rollout*.py`,
  `policy_worker*.py`, `run*.sh`), `REFERENCE-README.md`, `assets/camera-fit.json` (first, single-landmark fit).
* `cal2-calibration/`: scripts, the three reference JSONs, both recordings (`episode101.npz`, `episode120.npz`),
  logs, `wood-patch.png`, `inspect-background.png`; per replay `result.json`, `arrays.npz`, `contact.jpg`,
  `reference-configuration.json` and the frame-65 renders; for `replay-ep101-corrected-cameras` also the exported
  stages (`scene-reference.usda`, `synthetic-scene.usda`; they reference Spark asset paths) and the reference renders.
* `synthetic/`: scripts, README, `repeatability*.json`, `provenance.json`, the URDF he used.
* `calibration-evaluation/`: `replay_real_demo.py`, the replay configs, logs, summaries (old 212-episode dataset).
* policy evals (`act-step7000*`, `cal2-act-*`, `prompt-comparison*`, `smooth-control-comparison`,
  `eval-latest-gcp-6115`): scripts and result / evaluation JSONs only.

Not copied: `venv/`, `model/`, `hf-cache/`, `runtime/`, `studio/`, `robot-source/` (stock SO-ARM100 checkout),
`assets/converted/` (30 MB URDF-importer output of the stock URDF), `synthetic/raw|dataset|runs|plans*.json`
(2.3 GB + 0.75 GB + 0.4 GB of scripted-expert episodes), videos, checkpoints, most rendered frames.
