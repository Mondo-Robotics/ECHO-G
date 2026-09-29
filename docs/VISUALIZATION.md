# MuJoCo visualization

This is the previous G1 MuJoCo pipeline: **physical 39D prediction → motion NPZ →
MuJoCo MP4**. It consumes a model prediction directly and does not require GT motion.
The original physical export calculations, root trajectory reconstruction, camera
conventions and duration-preserving frame selection are retained. The public
renderer requires an explicit robot XML path and validates joint names instead of
assuming an installation-specific location or silently accepting another skeleton.

## Install and robot assets

```bash
python -m pip install -e '.[visualization]'
```

Obtain the G1 29-DoF MuJoCo model (`g1_mocap_29dof.xml`) and all referenced meshes
from the project's approved upstream robot asset distribution. Asset location and
revision instructions are **pending**. Robot XML/meshes are external inputs and are
not bundled in this branch. Preserve their relative directory layout. The model
must have a floating base, a `pelvis` body and the 29 canonical G1 hinge joints;
`left_hip_pitch_joint` and `left_hip_pitch` naming variants are accepted. The release
checks the expected 36 position coordinates and 35 velocity coordinates.

On a headless Linux GPU machine the renderer defaults to EGL. If the machine uses
a different supported MuJoCo graphics backend, set `MUJOCO_GL` and
`PYOPENGL_PLATFORM` before invoking the renderer. MP4 writing uses imageio-ffmpeg.

## Export and render

The prediction must already be in physical units at 30 FPS. It should contain
`robot_repr[T,39]`, `representation_units="physical"` and `fps=30`.

```bash
python scripts/export_robot_repr_mujoco_npz.py \
  --input results/v2_audio_text/seed_000/example.pt \
  --output results/v2_audio_text/mujoco/example.npz

python scripts/render_g1_motion.py \
  --npz results/v2_audio_text/mujoco/example.npz \
  --mjcf assets_external/unitree_g1/g1_mocap_29dof.xml \
  --out results/v2_audio_text/mujoco/example.mp4 \
  --fps 30 --width 960 --height 720 \
  --azimuth 180 --elevation -15 --distance 3
```

Installed entry points `echo-g-export-mujoco` and `echo-g-render` accept the same
arguments. For a directory of predictions, export with `--input-dir` and
`--output-dir`; relative subdirectories are preserved. Existing NPZ files require
`--overwrite` to replace them.

By default the first frame is grounded using G1 sole proxies, root velocity is
integrated, and every stored root orientation is used. Export alternatives are
`--root-translation-mode grounded_fixed`, `--root-orientation-mode first_frame`
or `identity`, and `--ground-each-frame`. These alter the visualization; record the
chosen modes and do not evaluate a modified rendering NPZ as if it were the original
prediction. The NPZ stores the chosen modes.

The camera follows the pelvis unless `--fixed-camera` is supplied. The front view
is `--azimuth 180`; `90` views the right shoulder. A 452-frame input at 30 FPS stays
452 frames at 30 FPS. Different rendering rates use nearest-frame sampling with a
rounded output frame count, preserving duration to the output-frame precision.

## Add the corresponding audio

The renderer produces a silent MP4. The dataset includes the matching clip-aligned audio at
`DATA_ROOT/audio/<stem>.wav`. To add it, set the duration to the actual output frame count divided
by its FPS (312 / 30 = 10.4 seconds in this example) and use a local FFmpeg installation:

```bash
ffmpeg -i results/v2_audio_text/mujoco/example.mp4 \
  -i data/v2/audio/example.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac \
  -af 'atrim=start=0:end=10.4,asetpts=PTS-STARTPTS' -t 10.4 \
  results/v2_audio_text/mujoco/example_with_audio.mp4
```

Use the clip-aligned audio, not an uncropped recording. The explicit audio interval starts at
zero and preserves the copied video stream. Check output stream durations with `ffprobe`; do
not use `-shortest` to silently remove video frames. Keep the silent source for frame extraction.

## Smoke check after dataset packaging

Select one released validation stem, generate its prediction from the new audited frozen
condition identity, and run the export/render commands above with the same external G1 XML and
meshes used in [the previous real render acceptance](BRANCH_VALIDATION.md). Use the new package's
audio for the mux step. Record the source dataset revision, condition/prediction hashes, robot
asset hashes, frame count, FPS, audio/video durations and output hashes. The clip
`english_1_wayne_0_100_100_utt_0000` has a frozen benchmark prefix of 312 frames at 30 FPS and is
suitable for this check. This is a packaging smoke test; the earlier render evidence does not
establish that a newly packaged dataset has passed it.

## Validation boundary

The release tests physical unit/FPS validation, first-frame grounding, integrated
translation, canonical joint schema, and duration-preserving frame selection on
synthetic motions. A real G1 render additionally needs the external robot asset
revision and a working graphics backend. Code-level tests do not establish that
arbitrary third-party XML geometry matches the benchmark skeleton.
