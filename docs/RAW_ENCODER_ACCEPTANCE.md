# Real raw-input acceptance — 2026-09-29

**Same-input reference parity passed. Historical frozen-cache parity did not pass.**
These are separate checks with separate conclusions.

The current V2 package ran three real BEAT2 WAVs and their original `meta.json` word times through
Wav2Vec2 extraction, Qwen extraction, merge, condition decoding, and best15k EMA motion inference.
The independent reference was the original V2 standalone encoder/inference implementation,
using exactly the same input files, encoder assets, and runtime.

[Machine-readable report](../validation/raw_encoder_acceptance.json) records per-tensor errors,
input identities, checkpoint/source SHA256 values, encoder weight/tokenizer identities, and all
pass/fail checks. Product source commit: `06055037f2bbdbf89a09868ad27d0c586d088a73`.

## Inputs and runtime

| Real clip | Audio duration | Output frames | Complete tokens |
|---|---:|---:|---:|
| `english_1_wayne_0_90_90_utt_0001` | 2.000 s | 60 | 8 |
| `english_21_ayana_0_26_26_utt_0000` | 19.710 s | 591 | 73 |
| `chinese_13_lu_2_66_66_utt_0000` | 19.610 s | 588 | 247 |

All source WAVs are 16 kHz mono. Word strings were joined canonically with spaces and complete
sequences were encoded without truncation. No frozen input features were supplied to the new
raw extraction path. The 247-token clip tests long text beyond the old 64-token capacity; it is
not part of the common3242 benchmark.

Runtime: NVIDIA A800-SXM4-80GB, PyTorch 2.7.0+cu128, Transformers 5.2.0, librosa 0.11.0,
SciPy 1.15.3, SoundFile 0.13.1; TF32 disabled. Qwen used BF16 and hidden layer −2; stored audio,
text, and token intervals are FP16. Missing optional Qwen fast-path libraries caused the
Transformers PyTorch fallback to be used for both implementations. No shared environment was
modified; installed dependencies passed import preflight before the run.

Motion inference used the actual V2 best15k EMA checkpoint, SHA256
`c1e7e863e28b76b710bc18f9e9029836771fc86c1bb9fca42aa9d7655f74a616`, seed 0, 8 Euler steps,
CFG=1, batch size 1, and checkpoint-local normalization. Predictions are finite physical39,
30 FPS, with lengths determined from their raw audio. Ground-truth motion files were not needed
by this inference path.

## Same-input reference comparison — passed

For all three clips, the following new-package outputs were bitwise equal to the independent
reference, with maximum absolute difference **0**:

- Frame-aligned acoustic features.
- Complete token features and pooled text.
- Word-derived token intervals and token count.
- Signed `frame_seconds - token_center_seconds` matrices; all include negative values.
- Final sampled and denormalized physical robot motion.

The merged conditions also passed the new provenance and input validation. This verifies the
actual encoders and raw-to-motion path on these inputs, beyond mocked-encoder or cached-input
tests.

## Comparison with the historical frozen cache — failed

The historical cache was compared separately. The retained thresholds are audio max error
0.001, text-token max error 0.5, pooled-text max error 0.25, exact token intervals, and physical
motion max/mean errors 0.05/0.002. They were not relaxed after observing these results.

| Clip | Audio max error | Text-token max error | Pooled-text max error | Motion max error | Motion mean error |
|---|---:|---:|---:|---:|---:|
| Wayne, 8 tokens | 0.001953125 | 0.250000 | 0.046875 | 0.001262188 | 0.000163586 |
| Ayana, 73 tokens | 0.019531250 | 0.375000 | 0.013916 | 0.001928866 | 0.000218050 |
| Lu, 247 tokens | 0.013427734 | 3.000000 | 0.054688 | 0.005267054 | 0.000424313 |

All three audio comparisons fail the original audio threshold. The 247-token text comparison
also fails its original threshold. Token intervals remain exact, and all three motion comparisons
pass the retained motion thresholds. Passing the motion thresholds does not cancel the failed
feature-cache checks. Physical39 mixes orientation, joint-angle, and velocity channels; the
motion errors above are elementwise representation errors, not a single distance in meters.

All six recorded Wav2Vec2 identity files and all nine Qwen configuration/tokenizer/weight files
match the SHA256 values from the 2026-09-19 raw validation record, including both Qwen weight
shards. The specific numerical cause has not been isolated; matching asset files alone does not
guarantee identical outputs across hardware/backend environments. The earlier report's exact
text-cache agreement cannot be generalized to this A800 run.

Use the frozen conditions for historical benchmark reproduction. This three-clip acceptance does
not establish a fresh-raw common3242 benchmark or MM20 result. [Reproduction scope](REPRODUCIBILITY.md)
retains the historical full-set results separately.
