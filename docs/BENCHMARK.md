# ECHO-G robot benchmark

Run `scripts/eval_g1_motion_cls.py` or the equivalent `echo-g-eval` entry point for G1 motion
metrics. Both call `echo_g.evaluation.g1_motion_cls`. The public benchmark requires robot
references, clip audio, the FGD encoder and the supplied BA normalization array.

## Install and assets

```bash
python -m pip install -e '.[benchmark]'
python -c "import librosa, scipy, soundfile; print(librosa.__version__, scipy.__version__, soundfile.__version__)"
```

The dataset is available in the private preview
[gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G); see the
[download guide](DATASET.md). Only authorized accounts can access it. ECHO-G model weights and
the FGD encoder have **not been uploaded**. Dataset archive and FGD/MMAE asset hashes are recorded in the
[dataset manifest](../manifests/dataset.json) and
[benchmark asset manifest](../manifests/benchmark_assets.json); this branch contains code,
protocols and identity records, not the large assets.
Use the same pinned asset revisions for every model in a comparison.

| Required input | Expected content | Distribution |
|---|---|---|
| Predictions | One physical `robot_repr[T,39]` tensor per `<stem>.pt`, 30 FPS | Generate with ECHO-G inference |
| References | Released ECHO-G robot motion files, same stems and coordinate system | [Private dataset preview](https://huggingface.co/datasets/gaopusen/ECHO-G) |
| Split | Released `splits/val_common.txt`; any formal exclusion list must be frozen and shared across methods | [Private dataset preview](https://huggingface.co/datasets/gaopusen/ECHO-G) |
| FGD encoder | Trained G1 skeleton-convolution AE `g1_aeskconv_full_pure2_w192.bin`, with its configuration/state | Hugging Face evaluation weights link: **pending** |
| BA normalization | `eval_assets/mmae/g1_mmae_30body_30fps.npy`, finite array of shape `(30,)` | Included in the [private dataset preview](https://huggingface.co/datasets/gaopusen/ECHO-G) |
| Audio | `audio/<stem>.wav`, matching the start of the released motion | Included in the [private dataset preview](https://huggingface.co/datasets/gaopusen/ECHO-G), under its source terms |

BA reads the included clip-aligned waveform, not the frozen acoustic feature tensor. The
loader accepts both `<wav-dir>/<stem>.wav` and the `<wav-dir>/<stem>/audio.wav` layout.
Do not shift the audio start, stretch it, or trim the stored file to the motion length; BA uses
the common evaluated prefix internally. Transcripts and word annotations accompany the data
but are not additional inputs to this motion evaluator.

The MMAE file is the frozen frame-weighted per-body central-difference speed normalization,
computed from the original 20,790-clip reference corpus (5,014,146 frames). It is a shared
benchmark reference, not the training split's 39D normalization statistics. Its SHA256 is
`0f763e6d54ca7bb33e49b0336ba2cc3359f9e9d5942e7f475c85c74b4b66e880`.
Do not recompute it on the 18,229 released clips: that would change the BA protocol.

Prediction payloads should include `representation_units="physical"` and `fps=30`.
Normalized network outputs must be de-normalized with the checkpoint's released
stats before evaluation. The evaluator performs no coordinate conversion or
normalization of the 39D motion representation.

## Evaluate one prediction per clip

Set the paths below to the downloaded dataset and external FGD encoder:

```bash
python scripts/eval_g1_motion_cls.py \
  --pred-dir results/audio_text/seed_000 \
  --ref-dir data/echo-g/motion_39d_30fps \
  --val-split data/echo-g/splits/val_common.txt \
  --wav-dir data/echo-g/audio \
  --mmae-file data/echo-g/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  --g1-ae-ckpt weights/g1_aeskconv_full_pure2_w192.bin \
  --fps 30 --ba-direction audio_to_motion \
  --enable-foot-metrics --require-all-stems \
  --tag audio_text_best --out results/audio_text/benchmark.json
```

Do not use `--limit` for a full benchmark. `--require-all-stems` fails on missing or
invalid clips or incomplete valid BA/jerk coverage. The archived source motion tensors
can be longer than the canonical common audio/motion prefix used for prediction: this
affects 433 of the 3,242 source validation clips. The command above preserves the
historical common-prefix evaluation and must cover 891,351 frames. It does not resample
motion or shift the start time.

For strict `--require-equal-lengths`, first create a separate reference directory whose
`robot_repr` tensors and `real_num_frames` are clipped to the supplied length CSV
`aligned_frames` (at most 600). Keep the original files and their identities. Then use
that directory as `--ref-dir` and add the flag; do not apply it directly to the longer
source tensors. The FGD encoder additionally uses complete multiples of 32 frames internally and
skips shorter clips for its features, matching the research implementation.
Store the command, checkpoint SHA, split SHA, evaluator source SHA and all evaluation
asset checksums beside the JSON. FGD is comparable only when the same G1 encoder is
used; it is not directly comparable to SMPL-X EMAGE paper values.

## MM20: 20 independent samples per clip

MM20 requires 20 generated predictions for each condition, using the **same model
weights and settings with different random seeds**. Copying one output 20 times
would measure zero diversity and is not a valid sampling experiment. Arrange the
physical predictions as:

```text
results/audio_text/mm20/
  run_000/<stem>.pt
  run_001/<stem>.pt
  ...
  run_019/<stem>.pt
```

Run the preceding benchmark command with these additional arguments:

```text
--multimodality-root results/audio_text/mm20
--multimodality-runs 20
--require-all-multimodality
```

All 20 samples for a clip must have exactly the same frame count. MM20 never crops
to GT, pads, resamples, or removes the tail. The primary value is `Multimodality`:
mean pairwise L1 distance in pose-only G1 FK positions, divided by `190` unordered
pairs and the frame count, then averaged equally over complete clips. It sums over
all 30 bodies and 3 coordinates rather than averaging the coordinates. The separate
`Multimodality_eq16_literal` diagnostic uses the denominator printed in
Audio2Gestures Eq. 16. Without repeated samples, the status is
`SKIPPED_NO_MULTIRUN` and the value is null, not zero.

## Metrics and aggregation

| JSON result | Meaning |
|---|---|
| `FGD` | Learned G1 gesture distribution distance; lower is better. Released pure2 encoder uses joints, excluding root motion. |
| `Div`, `Div_GT`, `Div_gap` | Within-clip position mean absolute deviation, frame weighted; compare absolute gap to GT. |
| `BA`, `BA_GT`, `BA_gap` | Audio-to-motion beat alignment; upper bodies, Gaussian width 0.3 s, frame weighted, no two-second head/tail trim. |
| `Jerk_length_weighted`, `Jerk_GT_length_weighted`, `Jerk_length_weighted_gap` | World-space jerk in m/s³, weighted by each clip's `T-3` valid third differences; compare gap to GT. |
| `Jerk`, `Jerk_GT` | Historical clip-equal jerk, retained separately for compatibility. |
| `foot_ground_error` | Mean absolute lowest sole-bottom height in meters, first-frame grounded anchor. |
| `contact_sliding_speed` | Horizontal stance-foot sliding speed in m/s. |
| `Multimodality` | MM20 diversity under the protocol above; interpret alongside quality metrics. |

BA, Div and MM20 use identity root orientation and zero translation. Jerk
uses stored root orientation and the trajectory integrated from local velocity and
yaw increments. Foot metrics use the same physical trajectory with an initial sole
grounding offset. Lower raw BA, Div or jerk is not automatically better: compare the
corresponding GT gap as well as the raw value.

## Source provenance

The packaged G1 encoder derives from the project's G1 adaptation of PantoMatrix
EMAGE evaluation code, with skeleton convolution from DeepMotionEditing and decoder
code from TM2T. Their source notices are retained in the vendored files. The rotation
utilities retain their PyTorch3D-derived attribution. No third-party model weights,
robot assets, SMPL-X assets or audio are included in this code branch. Final release
licensing and distribution details remain to be agreed by the project owners.
