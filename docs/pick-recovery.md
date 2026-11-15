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
| Missed grasps: the jaws closed below 2.4 (on nothing) after having been open (≥ 12) | 5th miss | `--max-misses` (0 = off) |
| Jaws shut on nothing, continuously, after having been open in this pick | 5 s | `--stuck-closed-s` (0 = off) |
| Time (unchanged) | 45 s | `--pick-timeout` |
| Failed picks in a row before the loop pauses for the operator | **5** (was 3) | `--max-attempts` |

`--miss-below 2.4` is set for the grasp residual (`--residual on`), which commands the gripper to 0 during the pick.
With it, empty jaws settle at 1.0–2.1 and a held brick at 2.9–3.5 (14 Nov 22:12 and 22:17 runs). Running **without**
the residual, empty jaws only reach 2.5–3.7 and a held brick reads 4.1–8.9, so pass `--miss-below 3.8` there.
Otherwise the rule never fires: it's safe but does nothing. A grasp confirmed on the same tick wins, because the grasp
check runs first. With the residual, the loop also needs the residual checkout's `hold_min 2.6` / `lost_below 2.4` to
see grasps at all; that change is separate from this one.

`--max-attempts` went up because a stop now costs about 16 s instead of 45 s. Five early stops still take less time
before a pause than three timeouts did, so the loop never asks for the operator sooner than before.

`--max-misses 0 --stuck-closed-s 0 --max-attempts 3` restores the old behaviour exactly.

## Evidence

**Residual on** (22:12 and 22:17 runs, 5 picks): no pick stopped, so held bricks at 2.9–3.5 never count. Only one
real empty close (2.0) was recorded, so catching residual misses still needs checking on the arm.

**Residual off, with `--miss-below 3.8`** (14 Nov traces, 67 picks, MolmoAct2 `ft_cal2_456_8k`):

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
