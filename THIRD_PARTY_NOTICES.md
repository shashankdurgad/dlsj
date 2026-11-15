# Third-party notices

The code in this repository is MIT-licensed (`LICENSE`). The parts listed below come from, or are derived from,
third-party projects and stay under their own licences. Datasets, models and the twin's fitted assets published on
Hugging Face are listed in [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md).

| Part | Where | Source | Licence |
|---|---|---|---|
| SO-101 URDF and meshes | fetched by `sim/twin/scripts/fetch_so101.sh` (not committed); vendored URDF `deploy/smolvla/twin_vendor/sim/twin/assets/so101/urdf/so101_new_calib.urdf`; the twin's generated robot USD | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 |
| Fin-ray (compliant) gripper finger CAD | `sim/twin/gripper/assets/SO101_soft_fin.stl` (unmodified); derived `finray_so101.npz` / `.json` (also vendored in `deploy/smolvla/twin_vendor/`) | [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) `hardware/SO101_soft_fin.stl`; see `sim/twin/gripper/assets/SOURCE.md` | Apache-2.0 (`sim/twin/gripper/assets/LICENSE`) |
| LDraw Duplo parts | `sim/twin/scene/assets/ldraw/` (unmodified) and the derived `sim/twin/scene/duplo_*.npz` | [LDraw.org Parts Library](https://library.ldraw.org/); authors listed in `sim/twin/scene/assets/ldraw/LICENSE.md` | CC BY 4.0 |
| DICE-RL learner | `sim/twin/rl/dice.py` is a port of the DICE-RL reference implementation (arXiv 2603.10263) | DICE-RL reference code, Copyright (c) 2026 Zhanyi Sun | MIT (text below) |
| Kinesthetic recording workflow | `dlsj/kinesthetic/` | [Flash ALOHA](https://github.com/JohnYao93/flash-aloha), Copyright (c) 2026 John Yao | MIT (text below) |
| MolmoAct2 | used as the base model of our fine-tune (not vendored); wrappers in `deploy/molmoact2/`, `sim/twin/policies/molmoact2.py` | [allenai/molmoact2](https://github.com/allenai/molmoact2), `allenai/MolmoAct2-SO100_101` | Apache-2.0, plus Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use) |
| Isaac Lab | `sim/twin/docker/build.sh` builds Isaac Lab's official `Dockerfile.base`; `sim/residual_env`, `sim/twin` import it | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | BSD-3-Clause (parts Apache-2.0) |
| Isaac Sim | the base image of the twin container | NVIDIA Isaac Sim | NVIDIA's Isaac Sim licence terms. **The Docker image is not redistributed**; only the Dockerfile and build script are in this repo. |
| Physical AI Studio, `physicalai` runtime, Anomalib, OpenVINO | dependencies (not vendored); `patches/anomalib-studio-437669f.patch` is a patch against the Anomalib Studio application | [open-edge-platform](https://github.com/open-edge-platform), [openvinotoolkit](https://github.com/openvinotoolkit) | Apache-2.0 |
| LeRobot | dependency (dataset format, recording, baseline) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | Apache-2.0 |
| Ultralytics YOLO | dependency of the brick detector / classifier training and inference code; `models/brick_in_view_v1/` and the YOLO release bundles are Ultralytics-trained weights | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 (or an Ultralytics enterprise licence). The YOLO weights are **not** covered by this repo's MIT licence. |

## DICE-RL reference implementation

```
MIT License

Copyright (c) 2026 Zhanyi Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Flash ALOHA

The kinesthetic recording workflow in `dlsj/kinesthetic/` (two-pedal go/reject keys, the review phase
that holds the arm after a take, replaying an unsaved take with fixed-timestep playback before deciding,
the scene-alignment overlay, and the saved start pose) is adapted from Flash ALOHA:
https://github.com/JohnYao93/flash-aloha

```
MIT License

Copyright (c) 2026 John Yao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
