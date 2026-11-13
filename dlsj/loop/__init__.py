"""Closed-loop controller: PERCEIVE -> DETECT -> REASON -> ACT -> OPTIMIZE (PLAYBOOK Section 8).

    skills.py     named pi0.5 skills (instruction string + termination conditions)
    executor.py   runs one skill: async action-chunk streaming at the control rate, safety, termination
    planner.py    task queue (phase 1: whatever brick is there; phase 2: ordered numbered bricks)
    reasoner.py   rule table + uncertainty handling (pure functions)
    controller.py the state machine tying it together, with per-stage timing and an event log
    telemetry.py  JSONL timing/event logs and median/p90/p99 summaries
    fakes.py      fake robot / cameras / inspector / scripted policy for dry runs and tests
    hardware.py   adapters for the real SO-101, CameraStream, the Anomalib inspector, policy back-ends
    status.py     browser status page (cameras, state, inspection heatmap, latencies)
    run.py        CLI: python -m dlsj.loop.run --dry-run

Only `dlsj.contracts` types cross module boundaries; LeRobot, OpenCV and OpenVINO are imported lazily.
"""
