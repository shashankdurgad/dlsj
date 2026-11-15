# Fin-ray gripper CAD

`SO101_soft_fin.stl` is copied unmodified from
https://github.com/Vector-Wangel/XLeRobot/blob/main/hardware/SO101_soft_fin.stl
(repository commit b017b5e6354bd9f61f4247a920c72622ca0aade0; file added 2025-08-11), licensed under the Apache
License 2.0 (`LICENSE` in this folder is the XLeRobot repository's licence file). It is the "Compliant Gripper (new)"
that TheRobotStudio/SO-ARM100 (Apache-2.0) links to under Optional Hardware: TPU 95A finger, PLA base, no supports,
two extra M3 screws.

`finray_so101.npz` / `finray_so101.json` are derived from it by `../build_assets.py` (the three bodies split, moved
into the SO-101 URDF link frames, convex collider pieces of the fin).
