# V2 robot benchmark

The release uses the current `eval_g1_motion_cls.py` implementation captured on
2026-09-28. Run it as `scripts/eval_g1_motion_cls.py` or `echo-g-eval`; both call
`echo_g.evaluation.g1_motion_cls`. Metric computations are preserved from the
research script. Internal paths have been replaced by explicit asset arguments.
Source and packaged-file SHA-256 hashes are in
[`scripts/benchmark_visualization_source_manifest.json`](../scripts/benchmark_visualization_source_manifest.json).

## Install and assets

```bash
python -m pip install -e '.[benchmark]'
python -c "import librosa, scipy, soundfile; print(librosa.__version__, scipy.__version__, soundfile.__version__)"
```

Dataset and model weights will be distributed on Hugging Face. Links and release
checksums are **pending**; this branch contains code and protocols, not these assets.
Use the same pinned asset revisions for every model in a comparison.

| Required input | Expected content | Distribution |
|---|---|---|
| Predictions | One physical `robot_repr[T,39]` tensor per `<stem>.pt`, 30 FPS | Generate with V2 inference |
| References | Released V2 robot motion files, same stems and coordinate system | Hugging Face dataset link: **pending** |
| Split | Released `splits/val_common.txt`; any formal exclusion list must be frozen and shared across methods | Hugging Face dataset link: **pending** |
| FGD encoder | Trained G1 skeleton-convolution AE `g1_aeskconv_full_pure2_w192.bin`, with its configuration/state | Hugging Face evaluation weights link: **pending** |
| BA normalization | `g1_mean_vel_30body.npy`, finite array of shape `(30,)` | Hugging Face evaluation assets link: **pending** |
| Audio | `<stem>/audio.wav` or `<stem>.wav`, matching the start of the released motion | Obtain/reconstruct from authorized upstream BEAT2 audio; instructions/link **pending** |
| SRGR semantics | Canonical 30 FPS cache, one `<stem>.pt` per clip | Hugging Face evaluation assets or reconstruction instructions: **pending** |

The latest evaluator requires semantic relevance data as well as audio. Each
semantic payload contains `stem`, `fps=30`, `num_frames`, `sem` (values in `[0,1]`),
and `protocol="beat2_official_first_match_30fps_v1"`. The cache must cover the full
evaluated prefix. It is never padded or resampled by the evaluator. Frozen text
conditioning features do not substitute for this semantic cache or the audio.

Prediction payloads should include `representation_units="physical"` and `fps=30`.
Normalized network outputs must be de-normalized with the checkpoint's released
stats before evaluation. The evaluator performs no coordinate conversion or
normalization of the 39D motion representation.

## Evaluate one prediction per clip

Set the paths below to locally downloaded/reconstructed assets:

```bash
python scripts/eval_g1_motion_cls.py \
  --pred-dir results/v2_audio_text/seed_000 \
  --ref-dir data/v2/motion_39d_30fps \
  --val-split data/v2/splits/val_common.txt \
  --wav-dir data/beat2_audio \
  --sem-dir data/benchmark/semantics_30fps \
  --mmae-file data/benchmark/g1_mean_vel_30body.npy \
  --g1-ae-ckpt weights/g1_aeskconv_full_pure2_w192.bin \
  --fps 30 --ba-direction audio_to_motion \
  --enable-foot-metrics --require-all-stems \
  --tag v2_audio_text_best --out results/v2_audio_text/benchmark.json
```

Do not use `--limit` for a full benchmark. `--require-all-stems` fails on missing or
invalid clips or incomplete valid BA/jerk coverage. The archived source motion tensors
can be longer than the audited common audio/motion prefix used for prediction: this
affects 433 of the 3,242 source validation clips. The command above preserves the
historical common-prefix evaluation and must cover 891,351 frames. It does not resample
motion or shift the start time.

For strict `--require-equal-lengths`, first create a separate reference directory whose
`robot_repr` tensors and `real_num_frames` are clipped to the frozen audit CSV
`aligned_frames` (at most 600). Keep the original files and their identities. Then use
that directory as `--ref-dir` and add the flag; do not apply it directly to the longer
source tensors. The 2026-09-29 package acceptance uses this explicit audited-prefix
reference preparation and checks every prediction length before evaluation.
The FGD encoder additionally uses complete multiples of 32 frames internally and
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
results/v2_audio_text/mm20/
  run_000/<stem>.pt
  run_001/<stem>.pt
  ...
  run_019/<stem>.pt
```

Run the preceding benchmark command with these additional arguments:

```text
--multimodality-root results/v2_audio_text/mm20
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
| `SRGR` | Semantic-mass-normalized positional recall at 0.1 m; a perfect prediction on a nonzero-semantic set scores 1. |
| `Jerk_length_weighted`, `Jerk_GT_length_weighted`, `Jerk_length_weighted_gap` | World-space jerk in m/s³, weighted by each clip's `T-3` valid third differences; compare gap to GT. |
| `Jerk`, `Jerk_GT` | Historical clip-equal jerk, retained separately for compatibility. |
| `foot_ground_error` | Mean absolute lowest sole-bottom height in meters, first-frame grounded anchor. |
| `contact_sliding_speed` | Horizontal stance-foot sliding speed in m/s. |
| `Multimodality` | MM20 diversity under the protocol above; interpret alongside quality metrics. |

BA, Div, SRGR and MM20 use identity root orientation and zero translation. Jerk
uses stored root orientation and the trajectory integrated from local velocity and
yaw increments. Foot metrics use the same physical trajectory with an initial sole
grounding offset. Lower raw BA, Div or jerk is not automatically better: compare the
corresponding GT gap as well as the raw value.

**Historical results:** this latest script includes semantic-mass-normalized SRGR
and length-weighted jerk. Do not attach an older table to a new evaluation protocol
without checking its metric definitions, encoder, selection and frame coverage.

## Source provenance

The packaged G1 encoder derives from the project's G1 adaptation of PantoMatrix
EMAGE evaluation code, with skeleton convolution from DeepMotionEditing and decoder
code from TM2T. Their source notices are retained in the vendored files. The rotation
utilities retain their PyTorch3D-derived attribution. No third-party model weights,
robot assets, SMPL-X assets or audio are included in this code branch. Final release
licensing and distribution details remain to be agreed by the project owners.
