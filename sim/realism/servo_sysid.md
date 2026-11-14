# SO-101 follower system identification (stage A1)

Source: the recorded teleop data, with `action` as the commanded target and `observation.state` as the measured position, both at 30 Hz.
- `duplo_teleop`: 517 episodes, old calibration. Split into 466 train and 51 held-out.
- `duplo_teleop_cal2`: 225 episodes, new calibration. Split into 203 train and 22 held-out.

Grasp and inspection frames come from the 3-phase labels. No `Present_Current` or load column was recorded.

Scripts and outputs:
- Script: `servo_sysid.py` (runs on the XPS, read-only).
- Simulator parameters: `servo_params.json`.
- Full numbers: `servo_sysid_raw.json`.
- Plots: `sysid_*_rollout.png`. Each shows a held-out episode with the command, the measured position and the model's open-loop prediction.

## Model

The best overall model is first order: `s[t+1] = s[t] + K·(a[t] − s[t])`, with **delay d = 0 ticks** in this indexing. The command recorded at tick t first shows up in the state read at t+1, so the effective lag is one control tick, 33 ms. Adding more pure delay made every joint worse.

The velocity-limited and second-order variants are within ±0.1° RMSE of plain first order:
- Second order is marginally better on wrist roll (and on shoulder pan in the cal2 set).
- Velocity-limited first order is marginally better on the gripper.

Since the gains are nearly equal, use first order with a velocity clip.

Held-out open-loop rollout RMSE in degrees, for the old calibration (cal2 in brackets):

| joint | K/tick | τ (s) | first-order | baseline: prev. command | baseline: hold |
|---|---|---|---|---|---|
| shoulder_pan | 0.302 (0.302) | 0.093 | **1.54** (0.52) | 3.49 (2.26) | 28.2 |
| shoulder_lift | 0.252 (0.259) | 0.115 | **2.38** (1.98) | 5.39 (4.01) | 80.8 |
| elbow_flex | 0.248 (0.249) | 0.117 | **1.82** (1.83) | 4.33 (3.42) | 61.6 |
| wrist_flex | 0.288 (0.301) | 0.098 | **0.61** (0.60) | 2.70 (2.24) | 26.5 |
| wrist_roll | 0.294 (0.320) | 0.096 | **0.54** (0.34) | 2.48 (1.58) | 22.4 |
| gripper | 0.205 (0.182) | 0.146 | **1.60** (2.32) | 3.39 (3.50) | 18.2 |

On every joint, the fitted model has 2–4× lower error than assuming "the measured position equals the previous command". The two calibration sets give the same dynamics, as expected, since only the encoder offsets differ.

## Envelope, bias and noise (old calibration)

| joint | v p99.5 (°/s) | v max (°/s) | a p99 (°/s²) | steady bias (°) | steady \|err\| p50 / p95 (°) | meas. noise σ (°) | teleop \|Δcmd\|/tick p50 / p95 / p99 / max |
|---|---|---|---|---|---|---|---|
| shoulder_pan | 169 | 311 | 1345 | −0.09 | 0.26 / 0.62 | 0.13 | 0.18 / 3.4 / 5.5 / 12.0 |
| shoulder_lift | 227 | 586 | 2057 | +0.66 | 0.66 / 6.37 | 0.15 | 0.53 / 4.9 / 7.1 / 15.1 |
| elbow_flex | 232 | 435 | 1662 | **+1.93** | 1.93 / 3.17 | 0.25 | 0.35 / 4.2 / 7.2 / 19.3 |
| wrist_flex | 169 | 388 | 1424 | +0.22 | 0.40 / 1.01 | 0.12 | 0.09 / 3.3 / 5.6 / 20.4 |
| wrist_roll | 161 | 322 | 1266 | −0.09 | 0.35 / 0.62 | 0.08 | 0.09 / 2.8 / 5.1 / 13.5 |
| gripper | 121 | 232 | 1488 | +1.23 | 1.23 / 2.75 | 0.11 | 0.00 / 3.3 / 6.6 / 16.3 |

- **Gravity sag:** at a steady command, the elbow sits about 1.9° off target, and shoulder lift's 95th-percentile error is 6.4°. These are the gravity-loaded joints. In Isaac, use finite drive stiffness, or add a gravity-direction bias term, rather than a perfect position servo.
- **Encoder noise:** about 0.1–0.25°, which is close to the STS3215's 0.088° per tick resolution. No deadband or backlash is visible beyond that.
- **Action rate:** teleop commands change by at most about 5–7° per tick at the 99th percentile. That's the realistic limit for a correction policy's output, and it matches the deployed controllers' `--max-step` 4–8.

## Gripper (513 old-calibration and 222 cal2 episodes with grasp/inspection labels)

| quantity | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| holding position, grasp+8 → inspection (0–100 units) | 4.34 | 5.16 | **5.75** | 5.89 | 6.15 |
| squeeze = command − measured while holding | −2.82 | −2.69 | **−2.55** | −1.76 | −1.07 |
| closing speed (most negative Δ per tick before the grasp) | −5.3 | −4.0 | **−3.8** | −3.5 | −2.6 |
| opening speed at release (max Δ per tick after inspection) | 2.6 | 3.4 | **4.0** | 4.1 | 5.9 |
| open level after release | 20.7 | 25.5 | **29.6** | 35.2 | 49.2 |

When holding a brick, the operator commands about 3.2 and the jaw stops at about 5.75 against the brick. That 2.5-unit gap is the squeeze. The open-loop model has no contact, so it closes all the way to about 3. The simulated brick's collision must stop the jaw at about 5.75, and the drive stiffness should produce a comparable holding force. cal2 is the same (hold 5.77, squeeze −2.94).

## Deployed-controller clamp counts (logs)

LeRobot's `max_relative_target` warnings per run:

| run | clamp warnings | most clamped joints |
|---|---|---|
| molmo/control.log | 574 | gripper 226, shoulder_lift 219, elbow 124 |
| smolvla/control.log | 1002 | gripper 506, shoulder_lift 404, elbow 171 |
| pi05_control.jitter.log | 5654 | shoulder_lift 2795, gripper 2330, elbow 1480 |
| pi05_control.prev.log | 668 | shoulder_lift 427, elbow 255 |
| pi05_control.0158.log | 140 | shoulder_lift 89, elbow 55 |

The policies regularly ask for larger per-tick jumps than the clamp allows, mostly on the gripper and shoulder lift. The simulated environment should apply the same per-tick clamp (`max_relative_target`: 10 for the π0.5 and SmolVLA runs, 20 in some) on top of the controller's `--max-step`. Otherwise a correction trained in simulation sees motions the real arm can't make. The latency regex only matched the two warm-up lines per log, so take latency from the controller's `/ctl/state` instead: about 340 ms for MolmoAct2, 74 ms for SmolVLA and 150–180 ms for π0.5 INT8.

## Using it in Isaac
1. **Control loop:** 30 Hz, with the command applied at the next tick (one-tick lag).
2. **Joint motion:** a first-order response with time constant τ ≈ 0.09–0.15 s per joint (or the equivalent PD stiffness and damping at the physics rate, tuned until the τ matches). Clip joint speed at the p99.5 velocity.
3. **Gravity sag:** a steady-state bias on the elbow (+1.9°) and shoulder lift (+0.7°, p95 6°), or let finite stiffness under gravity produce it.
4. **Measurement noise:** Gaussian σ of 0.1–0.25° on the observed joint state.
5. **Gripper:** contact stop at about 5.75 when holding a brick, a squeeze of about −2.5, and about 3.8 units/tick closing and 4 units/tick opening.
6. **Clamps:** the same per-tick clamp as the deployed controllers.
