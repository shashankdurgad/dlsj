# Showcase dashboard

Live 3D digital twin of the SO-101, mission control for the closed loop, and the presentation (data, training,
inspection, runtime optimisation, Physical AI Studio) on one page.

    ~/inference/deploy/showcase/run_showcase.sh        # on the XPS -> http://<xps>:8030/

- `server.py`: stdlib only. Relays `/api/state`, `/live/<cam>` and `/api/loop/<cmd>` to the loop controller
  (port 8020); serves `story.json` (the presentation content: every number measured, caveats included) and
  `replay.json` (a recorded run shown when the robot controller is off).
- `static/`: the page; `vendor/` = three.js 0.160 + urdf-loader 0.12.3 (bundled, works offline);
  `robot/` = SO-101 URDF + meshes from TheRobotStudio SO-ARM100 (Apache-2.0).
