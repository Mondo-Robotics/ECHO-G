# Benchmark

Use `scripts/eval_g1_motion_cls.py` (also available as `echo-g-eval`) for all four models.
Install the optional dependencies:

```bash
python -m pip install -e '.[benchmark]'
python -c "import librosa, scipy, soundfile; print(librosa.__version__, scipy.__version__, soundfile.__version__)"
```

## Inputs

[Download weights](../README.md#download-model-weights) and extract the
[dataset](https://huggingface.co/datasets/gaopusen/ECHO-G#download) into `data/echo-g`.
Use HF revision `v0.2.0`; [assets.json](../manifests/assets.json) pins every required asset.

| Input | Path |
|---|---|
| Physical predictions at 30 FPS | `results/audio_text/seed_000/<stem>.pt` |
| GT motion and split | `data/echo-g/motion_39d_30fps/`, `splits/val_common.txt` |
| Clip-aligned audio | `data/echo-g/audio/<stem>.wav` |
| BA normalization | `data/echo-g/eval_assets/mmae/g1_mmae_30body_30fps.npy` |
| FGD encoder | `weights/evaluation/g1_fgd_encoder.pt` |

Predictions must use physical units and the same coordinates as GT. BA reads waveforms,
not acoustic features; it uses the common evaluated prefix without shifting or stretching
the audio. Its fixed 30-body normalization comes from 20,790 reference clips / 5,014,146
frames. Recomputing it on the released training split would change the protocol.

## Full validation benchmark

Generate all 3,242 clips, without the quick start's `--limit`:

```bash
echo-g-sample --config configs/sgdit_audio_text.yaml --checkpoint weights/best.pt \
  --data-root data/echo-g --output-dir results/audio_text \
  --split val --seeds 0 --steps 8 --guidance-scale 1 --batch-size 3 --device cuda
```

For another model, use its [inference command](INFERENCE.md#dataset-clips), remove `--limit`
and update the prediction/output paths below.

```bash
python scripts/eval_g1_motion_cls.py \
  --pred-dir results/audio_text/seed_000 \
  --ref-dir data/echo-g/motion_39d_30fps \
  --val-split data/echo-g/splits/val_common.txt \
  --wav-dir data/echo-g/audio \
  --mmae-file data/echo-g/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  --g1-ae-ckpt weights/evaluation/g1_fgd_encoder.pt \
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

## MM20

Generate 20 independent samples of every condition with the same weights and settings.
The sampler writes `seed_000` through `seed_019`; the evaluator expects `run_000` through
`run_019`. Create aliases after sampling:

```bash
echo-g-sample --config configs/sgdit_audio_text.yaml --checkpoint weights/best.pt \
  --data-root data/echo-g --output-dir results/audio_text/mm20 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --steps 8 --guidance-scale 1 --batch-size 3 --device cuda
for i in {000..019}; do
  ln -s "seed_$i" "results/audio_text/mm20/run_$i"
done
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

## Reference scores

All results use 3,242 validation clips / 891,351 aligned frames at 30 FPS, EMA,
Euler 8 and CFG 1. Single-sample metrics use seed 0; MM20 uses seeds 0–19.

| Model / measurement | Selected step | FGD ↓ | MM20 |
|---|---:|---:|---:|
| Audio + text, released implementation | 15,000 | 2.278311 | Not rerun |
| Audio + text, historical checkpoint | 15,000 | 2.278349 | 1.785556 |
| Audio only, historical checkpoint | 25,000 | 2.360242 | Not reported |
| Text only, historical checkpoint | 15,000 | 2.436127 | 1.680639 |
| HumanRetarget after robot decoding, historical checkpoint | 25,000 | 4.724520 | 1.498274 |

Additional audio+text historical metrics:

| Metric | Prediction | GT | Absolute gap |
|---|---:|---:|---:|
| Div | 0.840263 | 1.159718 | 0.319455 |
| BA, frame weighted | 0.471369 | 0.534412 | 0.063043 |
| Jerk, weighted by T−3 (m/s³) | 48.322776 | 39.283955 | 9.038821 |
| Foot ground error (m) | 0.008435 | — | — |
| Contact sliding speed (m/s) | 0.051901 | — | — |

The released checkpoints preserve the source inference parameters and statistics.
Only the audio+text full-set FGD was rerun for the released implementation; the other
listed measurements remain historical results. Exact values and evaluation settings
are in [reference_results.json](../manifests/reference_results.json). Fresh feature
extraction or independent training may differ numerically; use the supplied frozen inputs.

## Attribution

The packaged G1 encoder derives from the project's G1 adaptation of PantoMatrix
EMAGE evaluation code, with skeleton convolution from DeepMotionEditing and decoder
code from TM2T. Their source notices are retained in the vendored files. The rotation
utilities retain their PyTorch3D-derived attribution. No third-party model weights,
robot assets, SMPL-X assets or audio are included in this code branch. The project-trained
FGD encoder weights are available in the public model repository under **CC BY-NC 4.0**.
The project's contribution to the frozen BA normalization is covered by the same data license.
Code and third-party source materials retain their applicable licenses; see
[NOTICE](../NOTICE).
