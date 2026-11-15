# Visual fit log (materials, textures, render settings)

Owner: the materials / look fit (`visuals/fit/`). Lights are the lighting agent's (`lighting.json`, `sim/twin/lighting/`),
camera response / blur the sensor agent's (`camera_color.json`, `sim/twin/sensors/`). Outputs:
`sim/twin/out/visual_fit/` (git-ignored).

## Tools
| script | what |
|---|---|
| `make_evalset.py` | 30 cal2 episodes x 5 moments (start / grasp / inspect / place / end), 2:1 fit / held-out split -> `evalset.{json,npz}` |
| `harness.py` | TwinEnv posed at every sample's recorded joint state, plates / bricks where the real frame shows them; region masks from the sim geometry |
| `render_eval.py --tag T` | render the whole set with the current look -> `render_T.npz`, `metrics_T.json`, `sheet_T_<cam>.jpg` |
| `eval_visuals.py` | the metrics: region Lab dE / highlight strength, sharpness, L histogram, SSIM, held-out patch classifier |
| `sweep_looks.py --looks L.json --tag T` | A/B of N candidate looks in ONE Isaac process (~30-60 s per look on 12 samples): region dE / dL / dC, highlight strength, per-plate-colour stats (+ linear means for `plate_albedo.py`), plate ring contrast, sheets + plate close-ups + renders npz |
| `plate_albedo.py` | joint plate albedo over the three cameras' responses (one albedo per plate, per-camera exposure gains) |
| `fit_visuals.py` | coordinate descent over material / light parameters (slow: ~40 s per evaluation; superseded by sweeps while the lights change) |
| `closeups.py` | real / before / after close-ups from two render npz |

## History
* tex1 / tex2 (previous session): table wood rebuilt (`wood_texture.py`: phone photo registered at 0.5 mm / texel, real knots and
  scuffs, quilted tile elsewhere, colours mapped to the top camera, normal map); inspect backdrop = sharp median of 40 episodes;
  hall cylinder from the panorama (people blurred, r 3 m); DLAA instead of DLSS (`render_settings.json`). Held-out top: sharpness
  sim 259 vs real 133, table dE 10.6, SSIM 0.59, classifier 0.83.
* mat1-3 (coordinate descent, old lights): stopped. Its only move was the table albedo x0.62, i.e. it compensated the exposure
  (the lighting agent's) - not a material property. Nothing written.
* s1 (old lights, 12 fit samples, top camera raw). Lacquer clear coat 0.5 / roughness 0.15 (unfitted) made big glare
  spots on the table (table highlight p99-p50 31.5 vs real 18.5, table dL +9). Softer looks: B (table cc 0.15 / ccr 0.35
  / rough 0.7; plates rough 0.4 / cc 0.1) 10.9, C (cc 0.25 / 0.25 / 0.6; plates 0.32 / 0.2) 12.7; warm tint
  (1, 0.95, 0.86) brings the table b* from 2.5 to 4.4 (real 3.2) and the wrist table b* to real. Wrote materials.json
  (cc 0.3 / 0.3 / 0.65, tint (1, .96, .89), plates 0.35 / 0.15 / 0.25) - superseded below.
* New light rig (lighting agent, 00:48: rect ceiling tubes ~11 m up, warm side lamps, dome share): s3 / s4h (12
  held-out samples). The table's top-view highlight stays ~30 (real 19) for clear coat 0.3 -> 0 AND base specular
  0.5 -> 0, so it is NOT the lacquer: the texture bakes the top camera's sheen (low frequencies of the top median) and
  the rig's diffuse falloff adds to it. Base specular adds a uniform veil: table dL +8.5 (spec 0.5) -> +7.1 (0.15) ->
  +6.1 (0). Chosen: table base specular 0.15 (new TwinLook.table_specular), clear coat 0.15 / roughness 0.35 (faint
  grazing sheen in the inspect view), roughness 0.8. Plates: the rig's plate highlights came out slightly weaker than
  real (lighting agent: area 2.1 % vs 3.2 %; s4h plate highlight 18.8 vs 22.6) -> plates a bit glossier: roughness
  0.22, clear coat 0.35 / 0.1.
* Plate albedo (plate_albedo.py on s4h, the camera responses of 00:49). No single albedo fits the three cameras:
  joint residual dE76 purple top 10.8 / wrist 13.2 / inspect 6.9, yellow 7.8 / 10.1 / 16.4, orange 7.6 / 7.4 (-).
  The real purple plate reads a* -6 / b* -32 in the top camera but a* +12 / b* -21 (lavender) in the wrist camera:
  that is the cameras' colour response (sensor agent). Per the plan (materials matched to the TOP camera; the
  wrist / inspect matrices are fitted on sim / real pairs), the albedos are the top-camera solution with the top
  exposure normalised by the table (g_top 0.75): orange (177, 101, 56), yellow (154, 180, 107), purple (67, 163, 233).
  A physically lavender purple albedo needs a top-camera colour matrix too (handed to the sensor agent).
* Plate rings (s3, s5h; plate close-ups sweep_s5h_plates.jpg). The visual plate's floor ripple was 0.25 mm (7 rings):
  clearly visible rings in the top view, where the real plates show none (their reflections are irregular, not
  ringed); at the inspect camera's grazing angle the real plates DO show rings in their reflections. The radial
  ring-contrast metric (sweep_looks.ring_energy; offline variant with sigma 2-3 px, radius 0.15-0.6) grows with the
  amplitude only in the top view (0 / 0.05 / 0.1 / 0.17 / 0.25 mm -> 0.06 / 0.04 / 0.06 / 0.15 / 0.40 vs real 0.49)
  and is flat in the wrist / inspect views; the real value is dominated by plate misregistration (real plate edge /
  bowl shading inside the sim mask), so it is an upper bound, not a target. Chosen by eye: 0.1 mm
  (assets.PLATE_VIS_RING_AMP, visual mesh only; the collider keeps plate_profile's 0.25 mm): rings barely visible from
  the top, faint in grazing reflections.
* Camera model with a top-camera response (sensor agent, 01:01; offset 0) - sweeps now render every camera through
  twin_scene.color_correct. Table "flat and bright" (lighting agent): s7x / s7y (2 x 3 held-out samples, the only
  size that ran while the GPUs were out of graphics-context memory, see below), lighting/metrics.contrast on the table:
  top luma mean / std / chroma real 117-119 / 32-33 / 11.4-12.0 vs sim 114 / 22 / 11.5-11.8; wrist std 28-30 vs 16-20;
  inspect std 43-45 vs 33-35. Mean and chroma already match in the top view; only the contrast is short. A texture
  tone curve (wood_tone.py; b = contrast 1.4, mean 0.9, chroma 1.3; c = 1.6 / 0.85 / 1.45) barely moves it (top std 22
  -> 24) while it darkens the mean to 100, over-saturates (chroma 15) and raises the top-view table highlight (32 ->
  38, real 18-20): the missing variance is low-frequency (light falloff / shadows / props on the real table), not the
  wood's contrast. Not applied (wood_tone.py kept as a tool; table_top_base.jpg = the untoned texture).
* Highlights (audit: wrist / inspect weaker than real). Clear coat table 0.6 / roughness 0.12, plates 0.7 / 0.04,
  bricks 0.4 / 0.05: inspect table highlight 31.9 -> 36.0 and 39.4 -> 42.2 (real 38.3 / 46.6); top table highlight
  unchanged (37.9 -> 37.6: the top view looks along the normal, where a clear coat reflects 4 %); top plates 12 -> 13
  (real 20). The wrist table highlight does not respond (19.6 -> 19.7 / 17.9 -> 14.9; real 22-24): the wrist camera
  looks steeply down, its highlights are the lights' (rig) reflections.
* Bricks (audit: too dark / grey): the per-colour brick stats (pixels where both real and sim show a saturated brick)
  are dominated by placement / label mismatch (a "light_blue" sim brick on a real pixel with a* +15 / b* +21); value
  x1.3 / chroma x1.2 made the wrist bricks far brighter than real (L 72-85 vs 48-68). Not changed; needs a dedicated
  measurement on detected brick crops (perception agent's detector).
* Inspect background (audit: the black laptop renders light): sim background luma p5 / p50 41 / 90 vs real 27-33 /
  65-75; gain 0.75 -> 36 / 82-86; backdrop_gain = hall_gain = 0.6 (luma ~ gain^(1/2.2): 90 x 0.6^0.45 = 72).
* GPU stalls (01:20-01:55): Isaac hung in PhysX set_transforms / the RTX frame wait after a few samples on both
  boxes (kernel: NVRM out of graphics-context memory with 8-9 Isaac containers per box). Not the bending fins (rigid
  fins hang too). sweep_looks.py gained --trace-s (stack dumps), --skip-eps / --moments / --offset and defaults to
  rigid fins (--physics-set). Cap since then: 4 Isaac containers per box; big render sweeps go to Modal
  (sim/twin/modal/fanout.py).

## Final (02:10, tag final_p0-4 = all 50 held-out samples in 5 short runs; materials-only A/B under the same light
rig and camera model; renders through each camera's response; merged: out/visual_fit/final_merged.json)
| camera | metric | old (tex2 materials) | final | real |
|---|---|---|---|---|
| top | table dE / dL / dC | 7.17 / +4.3 / 5.59 | 3.19 / -1.4 / 2.52 | - |
| top | table chroma | 8.5 | 11.4 | 11.3 |
| top | table highlight p99-p50 | 27.6 | 32.9 | 19.6 |
| top | plates dE / highlight | 9.31 / 16.6 | 8.20 / 13.4 | - / 22.3 |
| wrist | table dE / dL | 11.0 / +7.1 | 10.6 / +3.6 | - |
| wrist | plates dE | 12.6 | 12.2 | - |
| inspect | table dE / dL | 17.6 / +15.5 | 10.8 / +6.9 | - |
| inspect | table highlight | 26.3 | 32.7 | 42.8 |
| inspect | plates dE | 13.2 | 11.3 | - |
| inspect | background luma p5 / p50 | 36 / 84 | 28 / 70 | 24 / 69 |
| all | table luma std (top / wrist / inspect) | 23.7 / 15.3 / 39.9 | 22.7 / 14.8 / 34.2 | 33.6 / 28.3 / 47.2 |
Open: the table's luma spread (low-frequency, see above); top / wrist highlights weaker than real on plates and on
the wrist table (the rig's reflections; clear coat does not reach the steep views); the top "table highlight" is
higher because the warm, darker table lowers p50 under the texture's baked sheen (no glare spots in the sheets);
bricks / robot not changed (brick colour needs crop-level measurements); plate albedos were fitted before the top
camera got its root-polynomial response (02:00 sensor model) - refit them with that response (plate_albedo.py
supports linear M only).
