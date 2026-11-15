# Live twin mirror (`sim/twin/stream/`)

The real SO-101's pose, rendered live in the Isaac twin (realistic scene, fitted lighting, fin-ray gripper) from a hero
camera, streamed as MJPEG for an `<img>` embed.

```
sim/twin/stream/run_live.sh                                   # detached container dlsj-twin-live-<ts>, real pose feed
                                                              #   (http://127.0.0.1:8040/api/pose = the Mac's tunnel to the XPS)
sim/twin/stream/run_live.sh --pose_url http://127.0.0.1:8041/api/pose   # against the fake pose server
sim/twin/stream/run_live.sh --look minimal                    # black void, light matte arm, key + rim light
sim/twin/stream/run_live.sh --stop                            # stop + remove it
docker logs -f $(docker ps -q --filter name=dlsj-twin-live)

python3 sim/twin/stream/fake_pose.py --port 8041              # fake follower (stdlib; sinusoids inside the cal2 range)
sim/twin/run.sh sim/twin/stream/live.py --tune                # every named hero view, both looks -> out/stream/tune_*.jpg
sim/twin/run.sh sim/twin/stream/hero_search.py                # viewpoint search over real cal2 poses (~4 min)
```

## Endpoints (port 8030)
* `GET /stream.mjpg`: `multipart/x-mixed-replace; boundary=frame` JPEG stream, 960x540, up to 30 fps;
* `GET /health`: `{fps, render_ms, pose_age_s, frames, black_frames_skipped, pose_url, look, started, status, ...}`
  (`fps` = frames actually encoded over the last 5 s; `status` booting / live / stalled);
* `GET /frame.jpg`: the latest frame; `GET /`: a bare embed page. CORS `*` and no-cache headers everywhere.

**Bind:** `--bind 127.0.0.1` (default). Watson is a cloud box with host networking, so the stream only listens on
loopback; the Mac forwards watson `127.0.0.1:8030` to the XPS `localhost:8030` over SSH (and the XPS pose feed to
watson `127.0.0.1:8040`). `--bind auto` = the host's Tailscale IPv4 (`tailscale ip -4` / `tailscale0`) if present,
else 127.0.0.1, adding the Tailscale listener when it appears. Never `0.0.0.0` unless passed explicitly.
To look from the Mac: `ssh -F ~/.brev/ssh_config -L 8030:127.0.0.1:8030 watson`, then `http://localhost:8030/`.

## Input
`--pose_url` (default `http://127.0.0.1:8040/api/pose`) polled at 30 Hz: `{"t", "q": [pan, lift, elbow,
wrist_flex, wrist_roll, gripper], "frame": "cal2", "age"}`, LeRobot degrees in the cal2 frame, gripper 0-100; extra
fields are ignored. Until the first pose arrives the arm rests folded (`REST_DEG`); afterwards the last pose is held
when the server is unreachable, `q` is null, or `age` is null or > 1 s. Joints map to sim radians with the twin profile's W9 offsets (`ResidualEnv.to_sim`).

## How it works
* `live.py` (Isaac process) forks `server.py`'s child (pose poller, JPEG encoder with cv2, threaded HTTP server)
  before Kit boots; frames and poses go through shared memory (double-buffered RGB). PyAV is never imported.
* `TwinEnv` builds the scene (layout seed `--seed`, bricks / plates as the real data has them); its three robot
  cameras are swapped for one hero camera (`--hero`: a named view in `HEROES` or `ex,ey,ez,tx,ty,tz`, `--focal`).
  Default `hero` = `hero_a` from `hero_search.py`: 11 real cal2 poses (rest, mid-reach, 4 grasps at the deploy brick
  positions, lift, inspect, place over each plate); 360 cameras (azimuth 15 deg steps x elevation 15-50 deg x focal
  14 / 18 / 24 mm, looking at the working centroid, distance solved for a ~60 % arm height and backed off until no
  pose clips); the best 20 rendered (white-arm / hidden-arm / hidden-gripper renders = arm and gripper masks) and
  scored on clipping, height, gripper visibility, clutter behind the arm, centring, 3/4 angle and elevation.
  `hero_b`..`hero_f` are the runners-up (`--hero hero_c`); contact sheet in out/stream/hero_candidates.jpg.
* Background (`--look real`): the inspect backdrop billboard (a photo projected for the inspect camera) is hidden in
  the mirror; `--bg dark` (default) blacks out the hall panorama and the floor beyond the table edge, so the lit table
  reads as the real table in a dark room; `--bg hall` keeps the hall panorama and the blue floor. Only this process's
  stage changes; the twin's defaults are untouched.
  `--aa 3` (default) renders with DLSS quality (~22 fps at 960x540 on watson); `--aa 4` = the twin's fitted DLAA (~16 fps).
* Measured on watson (GB10, 2026-11-15), real look, 960x540: 21.7 fps over 60 s, render 36-50 ms, 0 black frames;
  pose polling: `--pose_conns 5` persistent HTTP/1.1 (keep-alive) connections, staggered, reconnecting on error;
  through the Mac's SSH tunnel one request takes ~150 ms (measured: 1 / 2 / 4 / 6 connections = 7 / 11 / 23 / 31
  requests/s), so 5 connections reach the 30 Hz cap.
* Each frame: critically damped filter (`--smooth_ms 100`) towards the latest pose -> `write_joint_state_to_sim`
  -> `sim.forward()` (link poses without a physics step, so bricks never move) -> VisualSync -> render.
  Frames whose max pixel is <= `--black_max` (GB10 black frames under GPU contention) are skipped.
* Robustness: the container runs with `--restart unless-stopped`; `live.py` exits (and is restarted) if the HTTP
  child dies or no non-black frame was rendered for 60 s.
