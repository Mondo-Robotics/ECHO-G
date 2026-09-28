# V2 reproduction protocol

This document distinguishes the original V2 experiment, its previously validated standalone
inference implementation, and the new `src/echo_g` package on this release branch. Historical
full-set results are reference targets, not proof that this branch has been rerun on the full
benchmark. Public data/weight downloads remain pending.

## Frozen model and recipe

The release model is **V2 audio+text, direct robot39, learned positions, QKNorm with global/local
Gaussian token attention**. Configuration: `configs/sgdit_v2_audio_text.yaml`.

| Setting | Reference value |
|---|---|
| Layers / hidden / heads / FFN | 12 / 768 / 8 / 2048 |
| Motion / audio / text dimensions | 39 / 1024 / 2560 |
| FPS / maximum valid frames / position capacity | 30 / 600 / 608 |
| Text capacity / largest audited source sequence | 256 / 247 |
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

[The model manifest](../manifests/v2_audio_text.json) records bounds, architecture, selection,
and the pending Hugging Face location. Hash the actual file before comparing results; a shared
filename or matching training step does not establish model identity. Original source-file
hashes identify the historical implementation, not the current package files.

## Dataset identity

| Source artifact | SHA256 |
|---|---|
| Full-text source release VERSION artifact | `610339f317ad3aa72a007cda14062cbdc39ca655930df14de1a1cb4b4007f4ad` |
| `splits/train_drop.txt` | `18c9d6abdc1eabc79889d2d2cfc6620e75cb2781db9e0824473411b8dbdfe66a` |
| `splits/val_common.txt` | `36a960134a321f7b2c47c1b24aabfdf1fa8ee213ea59a8570740be196ba7d83b` |
| `stats/drop_train.pt` | `9378214e1033a7fb309532a040b07f87a9010fbd484a30b8b9962cc75704aac8` |
| Source condition/motion audit CSV | `be4feba6a494d7175d5fc1243d17f4aa41d55d88fe9aa1a48866088b88130eb2` |

The source VERSION hash is not the hash of a future public archive. Public packaging will add
its own inventory, shard hashes, and Hugging Face revision. Counts and source identities come
from the original V2 experiment, not the old sinusoidal `train_all` experiment.

For inference, use complete frozen text states and the audited canonical lengths. V2 time
input is signed `frame_seconds - token_center_seconds`. Preserve the FP16 storage boundary for
token intervals before their FP32 center calculation. Do not substitute old 64-token caches,
absolute distances, B1 fixed priors, or a different normalization file.

## Historical results

The following scores are from the original V2 best15k model on common3242, 30 FPS, 891,351 frames,
EMA, 8 Euler steps and CFG=1. Single-sample metrics use seed000. They were recomputed from the
previous standalone inference port on 2026-09-19 using the historical frozen evaluator, commit
`e4be2e8955e47dc591427ae8d1d25cb4cc04de43`, script SHA256
`f76b3e281436eb0af13b3a76e70b7694e49067c6bfc2a77e080fe14cd64482e7`.

| Metric | Historical value |
|---|---:|
| FGD | 2.278349 |
| Div / GT | 0.840263 / 1.159718 |
| Absolute Div gap | 0.319455 |
| BA, frame weighted / GT | 0.471369 / 0.534412 |
| Absolute BA gap | 0.063043 |
| SRGR, semantic-mass normalized | 0.748832 |
| Jerk, clip-equal (m/s³) / GT | 50.683479 / 38.616035 |
| Jerk, weighted by T−3 (m/s³) / GT | 48.322776 / 39.283955 |
| Absolute weighted jerk gap (m/s³) | 9.038821 |
| Foot ground error (m) | 0.008435 |
| Contact sliding speed (m/s) | 0.051901 |
| MM20, separate seeds 0–19 run | 1.785556 |

Full precision, source report checksums, and separate single-seed/MM20 identities are preserved
in [v2_reference_results.json](../manifests/v2_reference_results.json). MM20 uses 64,840 generated
clips across 20 seeds; a seed000 run alone does not reproduce it. Generated jerk and Div should
be compared with GT rather than interpreted as monotonically better when smaller. Learned G1
FGD is comparable only under the same encoder and protocol, not directly to EMAGE paper scores.

This branch packages the latest `eval_g1_motion_cls.py`, with provenance recorded separately.
It has **not yet produced a new full common3242 benchmark with this packaged implementation**.
Use [BENCHMARK.md](BENCHMARK.md) for the current command and required assets. Retain the evaluator
and asset hashes beside future reports rather than relabeling this historical table as a new run.

## Validation scope

The new package passed five real-checkpoint canaries against the frozen original V2: input
tensors, nonzero flow velocities, eight-step Euler outputs, and physical denormalization were
exactly equal (maximum absolute difference 0), including 600 frames and 247 tokens. See
[BRANCH_VALIDATION.md](BRANCH_VALIDATION.md) for runtime, coverage, and the machine-readable report.
This branch has not rerun the full benchmark or raw encoder parity.

The earlier standalone V2 port reproduced all 3,242 predictions / 891,351 frames exactly against
the original model under identical frozen conditions and seed000. That is a historical result
for the earlier port. Tests or source parity in this release branch must be reported separately;
synthetic tests do not by themselves establish trained-checkpoint or full-dataset parity.

Fresh raw-input conditions are a separate reproduction mode. Earlier canaries matched historical
text states/times exactly and matched original versus ported motion exactly when both models
used the same newly extracted conditions. However, re-extracted audio differed from the old
cache by 0.001343–0.004639 maximum absolute error, exceeding the retained 0.001 tolerance on all
five canaries. Exact historical audio-cache reproduction is therefore not claimed. The full-set
reference scores above use frozen conditions; there was no full-set fresh-audio benchmark.

Fresh training can differ across PyTorch/CUDA/GPU environments. Use checkpoint/data identities,
recorded configuration, controlled input parity, and downstream scores to assess reproduction;
do not expect independently trained weights to be bitwise identical.

## Running the package

[README.md](../README.md) gives data validation, training/resume, canonical paired sampling,
independent cached inference, and raw condition extraction commands. Use the same config and
frozen inputs throughout a comparison. The dataset/weight manifests intentionally use null
Hugging Face fields until the actual uploads and immutable revisions exist.
