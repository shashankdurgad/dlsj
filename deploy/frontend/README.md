# Station display (port 8040)

A production-line "tablet" view of the cell: the live Isaac twin full screen, plus the loop, the silicon and the
output. Two files, no build step, no dependencies beyond Python 3's standard library.

- `server.py`: polls the running console, samples the XPS's CPU / GPU / NPU counters, keeps the brick ledger,
  relays the twin stream and the loop controls.
- `index.html`: the page (inline CSS and JS; Geist from Google Fonts, with a system-font fallback offline).

## Run it

```sh
# Anywhere, with no robot: a synthetic pick -> bring -> inspect -> place loop
python3 deploy/frontend/server.py --mock --port 8041        # open http://localhost:8041

# On the XPS, next to the robot
python3 ~/inference/deploy/frontend/server.py --console http://localhost:8020 \
    --twin http://localhost:8030/stream.mjpg --port 8040   # open http://localhost:8040 (or http://100.77.191.42:8040)
```

The server reads `index.html` on every request, so edits to the page show up on reload without a restart. Changes
to `server.py` need a restart.

## Where the data comes from

| Source | What | How |
|---|---|---|
| Console `:8020` | loop step, tallies, cycles, perception (bricks, plates, target, held brick), inspection, prompt, latency, measured joints, policy record | `GET /ctl/state` at 30 Hz. Works with `loop_control.py` (full loop) and `smolvla_control.py` / `run_finetuned.sh` (manual console, fewer fields) |
| Cameras | `top`, `inspect`, `wrist` MJPEG | the console's `/stream/<cam>`, loaded by the browser directly |
| Twin | Isaac live mirror on watson (`sim/twin/stream/`), realistic look, ~23 fps | reached on the XPS at `localhost:8030` through SSH tunnels on Desmond's Mac; relayed by this server as `/twin.mjpg` (+ `/twin/health`) |
| Silicon | CPU / GPU / NPU busy, 0-1, last 60 samples at 4 Hz | `/proc/stat`; xe GT idle residency `/sys/class/drm/card*/device/tile0/gt0/gtidle/idle_residency_ms`; intel_vpu `/sys/class/accel/accel*/device/npu_busy_time_us`. All readable without root. RAPL energy needs root, so there is no energy figure |
| Ledger | one entry per sorted brick: colour, bowl | built here: when a bowl's tally goes up, record the last sorted cycle's brick colour. The bowl implies the inspection mark (yellow = plain, orange = circle, purple = cross / uncertain). Resets when the loop restarts or this server restarts |

## Endpoints

- `GET /`: the page
- `GET /api/state`: the trimmed console snapshot + `link.age_s`, `now`, `silicon`, `ledger`
- `GET /api/pose`: `{"t", "q": [pan, lift, elbow, wrist_flex, wrist_roll, gripper], "frame": "cal2", "age"}`, the
  feed the twin renderer polls (LeRobot degrees, current follower calibration, gripper 0-100)
- `GET /api/config`: where the page finds the twin and the cameras
- `GET /twin.mjpg`, `GET /twin/health`: the relayed twin stream and its health
- `POST /loop/{start,pause,resume,stop,home}`: passed to the console's `/loop/<cmd>`

## The page

- Twin full bleed (the top camera stands in until the twin streams; a watchdog reattaches the stream when the
  renderer restarts or stalls).
- Top: model and steps, running status, Start / Pause / Home / Stop (Esc = Stop).
- Top right: the real camera next to its twin: the top view with YOLO boxes, plates and the target; the inspection
  view while the brick is at the camera.
- Bottom left: bricks per minute (sorted / minutes since the loop started), count sorted, inference ms.
- Bottom centre: Perceive / Pick / Inspect / Place, the active step tinted by the chip doing the work (GPU blue =
  the VLA moving the arm, NPU green = YOLO looking); below it the ledger of sorted bricks.
- Bottom right: live CPU / GPU / NPU lanes.
- An ambient glow in the colour of the brick in play.

## Design rules (from Desmond)

A production-line tablet: minimalist, elegant, useful. No narrative text or captions. The story (one Intel
laptop running the whole cell, each job on the right silicon, a twin mirroring reality, every brick inspected and
sorted) has to come through the layout and the live data. Black and white; brick and bowl colours only as data.
Add only what earns its place.
