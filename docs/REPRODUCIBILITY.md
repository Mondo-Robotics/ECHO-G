# ECHO-G reproduction protocol

Use the released checkpoint, frozen conditions and sampling settings below to reproduce the
reference benchmark. Fresh raw feature extraction and independent training can produce different
numerical results across environments.

## Frozen model and recipe

The release model is **ECHO-G audio+text, direct robot39, learned positions, QKNorm with global/local
Gaussian token attention**. Configuration: `configs/sgdit_audio_text.yaml`.

| Setting | Reference value |
|---|---|
| Layers / hidden / heads / FFN | 12 / 768 / 8 / 2048 |
| Motion / audio / text dimensions | 39 / 1024 / 2560 |
| FPS / maximum valid frames / position capacity | 30 / 600 / 608 |
| Text capacity / largest source sequence | 256 / 247 |
| Training data | Full-text release, `train_drop`, 14,987 clips |
| Validation split | `val_common`, 3,242 clips |
| Batch per GPU / gradient accumulation / effective batch | 3 / 16 / 48 |
| Optimizer / learning rate / weight decay | AdamW / 0.0003 / 0.0001 |
| Scheduler / warmup | CosineAnnealingLR over 63,000 optimizer steps / 0 |
| Precision | FP32; AMP and TF32 disabled |
| Conditional dropout | 0.1, jointly applied to audio/text/time conditioning |
| Temporal loss weight | 0.5 |
| EMA / gradient clipping | 0.999 / 1.0 |
| Training seed | 20260825 |
| Validation cadence | Every 5,000 optimizer steps and at completion |
| Checkpoint selection | Minimum EMA validation loss |
| Sampling | EMA, 8 Euler updates, CFG=1, stable stem/seed initialization |

The original run completed **63,000 optimizer steps**. The minimum EMA validation loss,
**0.289654246331365**, occurred at **15,000**; the final validation loss was
0.3519240033157446. The released reference checkpoint is best15k, not final63k and not an
FGD-selected checkpoint. Historical validation used `drop_last=True` and batch 3, covering
3,240 clips; full inference and the benchmark cover all 3,242.

## Checkpoint identity

Released filename: `best.pt`, from model repository
[gaopusen/ECHO-G](https://huggingface.co/gaopusen/ECHO-G/tree/2026-09-30), revision `2026-09-30`.

```text
Release SHA256: c84f31fce140cdc8c3be5e5dbc8cb8eb3a82ce1865e2d84026a0791d4735928d
Source SHA256:  c1e7e863e28b76b710bc18f9e9029836771fc86c1bb9fca42aa9d7655f74a616
step: 15000
weights: EMA
```

The published checkpoint removes unrelated training metadata and internal paths while
preserving every inference parameter and normalization value. Strict loading, parameter
equality and seven real-clip prediction/FGD-feature comparisons passed; the maximum absolute
output difference was 0. These checks establish packaging equivalence and do not constitute
a new full-split benchmark. See the [model card](MODEL_CARD.md) for both published weight files.

[The model manifest](../manifests/audio_text.json) records the checkpoint and pinned
Hugging Face location. Dataset details and download instructions are in the
[data card](DATASET_CARD.md) and [download guide](DATASET.md).

## Data and timing

Use the supplied `train_drop` / `val_common` splits and `stats/drop_train.pt` without recomputing
them. The ECHO-G configurations already select the released condition metadata and training stats.

For inference, use complete frozen text states and the canonical frame lengths. ECHO-G time
input is signed `frame_seconds - token_center_seconds`. Preserve the FP16 storage boundary for
token intervals before their FP32 center calculation. Use the complete token sequence, signed
time distances, learned prior parameters, and supplied normalization statistics.

## Reference results

The following scores are from the reference checkpoint on common3242, 30 FPS, 891,351 frames,
EMA, 8 Euler steps and CFG=1. Single-sample metrics use seed000.

| Metric | Historical value |
|---|---:|
| FGD | 2.278349 |
| Div / GT | 0.840263 / 1.159718 |
| Absolute Div gap | 0.319455 |
| BA, frame weighted / GT | 0.471369 / 0.534412 |
| Absolute BA gap | 0.063043 |
| Jerk, clip-equal (m/s³) / GT | 50.683479 / 38.616035 |
| Jerk, weighted by T−3 (m/s³) / GT | 48.322776 / 39.283955 |
| Absolute weighted jerk gap (m/s³) | 9.038821 |
| Foot ground error (m) | 0.008435 |
| Contact sliding speed (m/s) | 0.051901 |
| MM20, separate seeds 0–19 run | 1.785556 |

MM20 uses 64,840 generated clips across 20 seeds; a seed000 run alone does not reproduce it.
Generated jerk and Div should be compared with GT rather than interpreted as better whenever
smaller. G1 FGD is comparable only with the same encoder and protocol.

The packaged implementation scores **FGD 2.278311** on the same 3,242 clips / 891,351 frames
using the released evaluator. The MM20 value above remains from the original 20-seed run.
Use [BENCHMARK.md](BENCHMARK.md) for metric definitions and commands.

## Numerical reproducibility

Use the supplied frozen conditions for benchmark reproduction. Re-extracting audio or text
features can change their values with encoder revisions, decoding, hardware and numerical
backends; exact agreement with the historical feature cache is not guaranteed. Independent
training can likewise differ across PyTorch, CUDA and GPU environments.

The [README](../README.md) gives training, sampling and raw condition extraction commands.
Keep the checkpoint, dataset, configuration, sampling seeds and evaluation assets fixed when
comparing results.
