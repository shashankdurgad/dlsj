# Twin lighting: fit log

Owner: lighting (`sim/twin/lighting/`, the light fields of `visuals/lighting.json`, the light code of
`visuals/twin_scene.py`, the light section of `env/randomize.py`).

## Pipeline
```
NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/basis_render.py --tag v1     # one-light renders, 60 samples (~16 min)
sim/twin/run.sh sim/twin/lighting/fit_rig.py --basis v1                             # NNLS light weights (no Isaac, ~10 min)
NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/calib_units.py               # RTX units of rect / sphere / spot lights
sim/twin/run.sh sim/twin/lighting/make_rig.py [--out ...]                            # writes the rig into visuals/lighting.json
NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/validate.py --rig <json> --tag vN   # old vs candidate, held-out
NAME=dlsj-twin-light sim/twin/run.sh sim/twin/lighting/dr_grid.py --tag v1           # lighting DR at scale 0 / 0.5 / 1
python sim/twin/lighting/test_spotlights.py                                         # DR draw checks (plain Python)
```
Evaluation states: visual-fit's evalset (`out/visual_fit/evalset.{json,npz}`; recorded joint states and real
layouts, 5 moments per episode). Fit episodes 2 49 95 142 189 235 282 329, held-out 33 126 220 313 (20 samples).
Metrics: `metrics.py`, invariant to per-image gain and white balance (the RealSenses ran AE / AWB).

## Physical facts (user)
* Ceiling: fluorescent strips ~10 m above the table, weak; long axis assumed parallel to the table's long edge.
* Warm lamps (2700-3200 K) ~6 m out along the inspect camera's view direction; their bloom is sometimes visible.
* Cameras on auto exposure and auto white balance (sensor agent models them).
* Verdict on the first twin look: lights too strong and too hard, real highlights faint and elongated.

## Findings
1. **RTX flat ambient.** Isaac Lab sets `/rtx/sceneDb/ambientLightIntensity` 1.0 (colour 0.1): with every light off
   the top table still renders at sRGB 57.8 of 133.5 (linear ~18 % of the light), unoccluded, so shadows wash out.
   The rig sets it to 0 (`ambient_intensity`) and the dome carries the ambient light.
2. **Units.** RTX renders sphere, rect and shaped (spot) lights at 0.295 x the textbook radiance formula relative to
   a distant light of the same nominal irradiance, the same for all three types and sizes (`calib_units.py`,
   `out/lighting/units.json`); `rig.RTX_AREA_UNIT` corrects it. A dome of intensity I lights the table like a
   zenith distant light of 0.56 I (`rig.DOME_TO_DISTANT`, light basis).
3. **Room stand-ins block light.** The unlit hall cylinder (r 3 m, up to 2.6 m) and the inspect backdrop are opaque
   to lights: distant lights below ~41 deg elevation rendered nothing, and lamps 6 m out were blocked. The rig sets
   `primvars:doNotCastShadows` on `HallVisual` / `BackdropVisual` (`rig.no_shadows`).
4. **Real shadows.** Mid-episode / start-frame ratio maps (`metrics.shadow`): the arm and the held brick cast soft
   shadows that fall to image +x +y in the top view = env +x +y, i.e. light from behind the robot (env -x, -y); real
   shadow depth 0.24 (top) / 0.31 (inspect). Debug: `scratchpad .../interim_real_shadow_maps.jpg`.
5. **Light-basis fit** (`fit_rig.py`, per-channel NNLS with free per-frame gains): the top camera prefers overhead
   light from az -170 / el 64 and az 135 / el 60 (and some from az 90 / el 46), i.e. behind the robot. That agrees
   with visual-fit's plate-highlight clusters (az ~125 / el ~62, az 180 / el 50-65) and the shadow direction. The
   fitted colours per cluster are not physical (they absorb material colour errors) and the full-image fit improves
   the held-out relative RMSE only from 0.369 (dome only) to 0.345 (top). The images are dominated by
   material / texture / camera mismatch, so the rig takes DIRECTIONS from the fit and the highlights, and SHARES from
   the user's verdict plus the metrics below, not the raw NNLS weights.
6. **Inspect view.** The real far table shows a warm grazing sheen (lacquer reflecting low, warm sources in front of
   the inspect camera); the old twin had none (highlight area 0.09 % vs real 4.5 %). The lamps' mirror point lands
   ~0.6 m in front of the inspect camera (on the far table, as in the real frames) for a lamp height of ~1 m at 6 m;
   at 2.5 m it lands right in front of the camera. The real left-third blue / purple cast (log B/G +0.23..0.38 vs
   the middle) is the lacquer reflecting the purple-lit hall: no light placement changed it (tint error 0.47 -> 0.45).

## Nominal rig (visuals/lighting.json, written by make_rig.py)
Shares of the nominal top-table brightness; the total is the old rig's, so the visual fit's exposure calibration
holds. Colours are divided by their share-weighted mix (scene average stays white for the AWB-fitted materials).

| light | prim | what | share |
|---|---|---|---|
| dome | /World/Light | broad hall / ceiling light, 5000 K | 0.745 |
| key | /World/Sun | the ensemble of overhead strips: distant, 40 deg wide, az -170 / el 72, 4500 K + green | 0.08 |
| 2 strips | /World/TwinCeil0-1 | RectLight 4.5 x 0.25 m, 11.3 m out (10 m up), az -170 / el 62 and az 128 / el 62, long side along the table edge | 2 x 0.025 |
| 3 far strips | /World/TwinCeil2-4 | RectLight 2.4 x 0.25 m, 12 m out at el 12, inspect view az +-35 deg (grazing sheen) | 3 x 0.015 |
| 2 warm lamps | env_i/TwinSide0-1 | SphereLight r 0.25 m, 3000 K, 1 m up, 6 m out along the inspect view, +-1.5 m sideways | 2 x 0.06 |
| RTX ambient | carb | off | 0 |

## Metrics (held-out, 20 samples; old = the twin before this work, i.e. visual-fit's dome + 12 deg key + 3 compact 4 deg ceiling lights + RTX ambient)
| camera | metric | real | old twin | new rig (v2) |
|---|---|---|---|---|
| top | shadow area | 0.037 | 0.014 | 0.014 |
| top | shadow depth | 0.236 | 0.214 | 0.239 |
| top | highlight plate/sheen area | 0.032 | 0.025 | 0.021 |
| top | shadow iou | | 0.118 | 0.142 |
| top | shadow dir_err_deg | | 63.949 | 46.269 |
| top | highlight pos_err | | 0.532 | 0.493 |
| top | tint error (L/R chroma) | | 0.067 | 0.076 |
| top | brightness grid rms | | 0.638 | 0.633 |
| inspect | shadow area | 0.142 | 0.037 | 0.036 |
| inspect | shadow depth | 0.313 | 0.189 | 0.231 |
| inspect | highlight plate/sheen area | 0.044 | 0.001 | 0.027 |
| inspect | shadow iou | | 0.112 | 0.080 |
| inspect | shadow dir_err_deg | | 11.075 | 19.877 |
| inspect | highlight pos_err | | 1.200 | 1.533 |
| inspect | tint error (L/R chroma) | | 0.468 | 0.453 |
| inspect | brightness grid rms | | 1.030 | 1.098 |

v3 (strips 3x dimmer radiance: 4.5 m long, half the irradiance; dome 0.66 -> 0.745, key 0.12 -> 0.08), against the
installed v2 rig, AFTER visual-fit's material update (lacquer cc 0.3 / ccr 0.3 / rough 0.65, plates rough 0.35):
| camera | real | v2 rig | v3 rig |
|---|---|---|---|
| top plate_strength (p99-p50 L*) | 21.5 | 19.0 | 18.8 |
| top table_strength (p99-p50 L*) | 16.4 | 27.2 | 27.2 |
| top plate highlight area | 0.032 | 0.013 | 0.013 |
| inspect plate_strength (p99-p50 L*) | 29.6 | 36.9 | 36.6 |
| inspect table_strength (p99-p50 L*) | 40.5 | 35.1 | 35.3 |
| inspect plate highlight area | 0.044 | 0.042 | 0.043 |

The strips no longer drive the top-view highlights (3x dimmer strips: plate strength 19.0 -> 18.8, table 27.2 -> 27.2,
real 21.5 / 16.4); the sim plates are now slightly LESS glossy than real. The top table's p99-p50 above real is
texture contrast (dark scuffs in the rebuilt wood), not glare: no glare patch is left in compare_v3_top_crops.jpg.

## Lighting DR (randomize.py light section = spotlights.py)
See spotlights.py for ranges. One knob: `TwinRandomizer.set_light_scale(s)` / `cfg["light"]["scale"]`.
Grid: `out/lighting/dr_grid_v1_{top,inspect}.jpg` (rows: 3 samples x scale 0 / 0.5 / 1; real | nominal | 4 draws).
Spot strengths were 3x too strong in that grid (the nominal table irradiance used pi x dome intensity; fixed with
DOME_TO_DISTANT after the grid).
Final grid with the corrected spot units: `out/lighting/dr_grid_v2_{top,inspect}.jpg`.

### Interface for rl/ (and anyone resetting with DR)
```
from sim.twin.env.randomize import TwinRandomizer
dr = TwinRandomizer(cfg)          # cfg["light"] = partial overrides of spotlights.LIGHT_DR (deep-merged)
dr.set_light_scale(s)             # 0 = nominal fitted rig, 1 = full range; takes effect at the next reset / draw
draw = dr(env, rng)               # draw["light"]: exposure, per-light gain / rgb / dir / pos, off, dome_texture,
                                  #   spots[env] = [{pos, aim, cone_deg, softness, radius, strength, rgb, hue}]
dr.restore(env)                   # nominal again (spots dark, lights / poses / colours restored)
```
rl/visual_dr.Ramp.apply already calls `set_light_scale` after replacing the cfg; its `scaled_cfg` shrinks only the
light keys it knows (`exposure` tuple, `p_dome_texture`), everything else follows the light scale.

## Open items
* Real cast-shadow AREA is 2.5x (top) / 4x (inspect) the sim's at matched states, at similar depth. The lights are
  soft and from the right side, so the remaining gap is probably geometry (cables / hands / arm silhouette) plus the
  ~10 px sim-real camera offset that the metric's margins remove; not pursued further.
* Inspect left-side blue / purple cast: lacquer reflecting the purple-lit hall (hall texture x table gloss), not a light.
* The far strips and lamps make the warm grazing sheen in the inspect view (area 0.09 % -> 2.7-4.3 %, real 4.5 %);
  its shape is a broad glow, the real one streaky (the real lacquer is uneven).

## Top-view haze check (replay ep122, 00:56)
`contrast_check.py` / `metrics.contrast` (table / plate luma mean, std, Lab chroma; no AE / AWB normalisation).
* The replay's sim top frames have luma std 13 and a black level (p2) of 92 from frame 0 on (real: std 38, p2 8).
  Our validate renders of the same rig (static, 3 passes, camera response applied) have std 31.5, p2 48.
* p2 = 92 is exactly sRGB(0.116 linear): the top camera model in visuals/camera_color.json at 00:56 had
  `offset_linear` [0.116, 0.110, 0.096] (a veiling-glare offset), which lifts black to grey everywhere. The sensor agent
  set it to 0 at 01:01. Not the lights.
* Lights: rig v2 vs v3 (dome share 0.66 vs 0.745) and v3 vs v3 + dome_specular 0.2 change table luma std by < 0.5
  and chroma by < 0.2 (validate_v3 / v4): the lighting is not what flattens the image. `dome_specular` (rig.apply_dome)
  is available but left unset.
* Remaining real-vs-sim gap at matched states (v4, table region): luma std 24 vs 33, chroma 7.8 vs 9.8, mean 132 vs
  118. Under near-uniform light the table's contrast is the texture's (wood rebuild) and the material's: visual-fit.
