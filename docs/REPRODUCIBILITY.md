# V2 reproduction protocol

Use the released checkpoint, frozen conditions and sampling settings below to reproduce the
reference benchmark. Fresh raw feature extraction and independent training can produce different
numerical results across environments.

## Frozen model and recipe

The release model is **V2 audio+text, direct robot39, learned positions, QKNorm with global/local
Gaussian token attention**. Configuration: `configs/sgdit_v2_audio_text.yaml`.

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

Reference filename: `best.pt` (suggested local name: `v2_best015000.pt`).

```text
SHA256: c1e7e863e28b76b710bc18f9e9029836771fc86c1bb9fca42aa9d7655f74a616
step: 15000
weights: EMA
wordtime schema: release30-bounded-qk-global-local-wordtime-v2.1
```

[The model manifest](../manifests/v2_audio_text.json) records the checkpoint and pending
Hugging Face location. Dataset details and download instructions are in the
[data card](DATASET_CARD.md) and [download guide](DATASET.md).

## Data and timing

Use the supplied `train_drop` / `val_common` splits and `stats/drop_train.pt` without recomputing
them. The V2 configurations already select the released condition metadata and training stats.

For inference, use complete frozen text states and the canonical frame lengths. V2 time
input is signed `frame_seconds - token_center_seconds`. Preserve the FP16 storage boundary for
token intervals before their FP32 center calculation. Do not substitute old 64-token caches,
absolute distances, B1 fixed priors, or a different normalization file.

## Historical results

The following scores are from the original V2 best15k model on common3242, 30 FPS, 891,351 frames,
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
