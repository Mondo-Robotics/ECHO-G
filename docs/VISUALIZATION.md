# MuJoCo visualization

The G1 MuJoCo pipeline is: **physical 39D prediction → motion NPZ →
MuJoCo MP4**. It consumes a model prediction directly and does not require GT motion.
It reconstructs the physical root trajectory, uses duration-preserving frame selection, and
validates the joint names in the supplied robot XML.

## Install and robot assets

```bash
python -m pip install -e '.[visualization]'
```

Follow [the HF dataset instructions](https://huggingface.co/datasets/gaopusen/ECHO-G#download) to extract the robot-asset archive alongside
the data. It supplies `assets/unitree_g1/g1_mocap_29dof.xml`, the referenced meshes and the
BSD-3-Clause license. Preserve their relative directory layout. The model
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
echo-g-export-mujoco \
  --input results/audio_text/seed_000/example.pt \
  --output results/audio_text/mujoco/example.npz

echo-g-render \
  --npz results/audio_text/mujoco/example.npz \
  --mjcf data/echo-g/assets/unitree_g1/g1_mocap_29dof.xml \
  --out results/audio_text/mujoco/example.mp4 \
  --fps 30 --width 960 --height 720 \
  --azimuth 180 --elevation -15 --distance 3
```

For a directory of predictions, export with `--input-dir` and `--output-dir`; relative
subdirectories are preserved. Existing NPZ files require `--overwrite` to replace them.

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
ffmpeg -i results/audio_text/mujoco/example.mp4 \
  -i data/echo-g/audio/example.wav \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac \
  -af 'atrim=start=0:end=10.4,asetpts=PTS-STARTPTS' -t 10.4 \
  results/audio_text/mujoco/example_with_audio.mp4
```

Use the clip-aligned audio, not an uncropped recording. The explicit audio interval starts at
zero and preserves the copied video stream. Check output stream durations with `ffprobe`; do
not use `-shortest` to silently remove video frames. Keep the silent source for frame extraction.
