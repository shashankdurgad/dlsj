# Pick recovery: missed grasps → go home → try the prompt again

The VLA picks well on its first try, but once it misses it tends to keep fumbling where it is. Before this change a
pick only ended on a confirmed grasp or after `--pick-timeout` (45 s). Now the loop also stops a pick that is going
nowhere, goes home and runs the same pick prompt again. That's the path the 45 s timeout already used: `attempts += 1`,
`_go_home`, then survey → pick.

Code: `deploy/smolvla/pick_watch.py` (pure, about 30 lines), called from `_step_pick` in `loop_control.py` while the
VLA runs the pick. Picks still start wherever they start today (from above the plate after a place). Only a failing
pick goes home.

## When a pick is stopped

| Rule | Default | Flag |
|---|---|---|
| Missed grasps: the jaws closed below 3.8 (on nothing) after having been open (≥ 12) | 5th miss | `--max-misses` (0 = off) |
| Jaws shut on nothing, continuously, after having been open in this pick | 5 s | `--stuck-closed-s` (0 = off) |
| Time (unchanged) | 45 s | `--pick-timeout` |
| Failed picks in a row before the loop pauses for the operator | **5** (was 3) | `--max-attempts` |

`--miss-below 3.8`: closed on nothing, the jaws bottom out at 2.5–3.7. On a brick they stall at 4.1–8.9. A
grasp confirmed on the same tick wins, because the grasp check runs first.

`--max-attempts` went up because a stop now costs about 16 s instead of 45 s. Five early stops still take less time
before a pause than three timeouts did, so the loop never asks for the operator sooner than before.

`--max-misses 0 --stuck-closed-s 0 --max-attempts 3` restores the old behaviour exactly.

## Evidence (14 Nov XPS traces, 67 picks, MolmoAct2 `ft_cal2_456_8k`)

Replaying every recorded pick through `PickWatch`:

| Picks | Count | Stopped by the new rules |
|---|---|---|
| Clean successes (≤ 10 s) | 43 | **0** |
| Slow successes after fumbling (13–35 s) | 11 | 6 (those taking 19–35 s), sent home to try again |
| Failures (timeout or stopped) | 13 | 10, after a median of 15.7 s instead of 45 s (7 misses, 3 stuck closed) |

A miss happens about every 2.5 s, and clean picks never had more than 2. The 6 slow successes are the trade-off: they
go home and retry instead of fumbling on. The traces can't say how often that retry wins, so check it in
`loop_*.jsonl`: the `pick_miss` and `pick_stop` events, and the cycle result `pick stopped: misses | stuck_closed`.

Re-check the 3.8 threshold if the gripper calibration or the policy changes. The grasp residual (`--residual on`) was
not running in these traces.
