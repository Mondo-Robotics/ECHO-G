# Data format

The [HF dataset](https://huggingface.co/datasets/gaopusen/ECHO-G) provides download instructions,
source attribution and the full dataset card. Use the supplied split lists: 14,987 training
and 3,242 validation clips. Stable clip stems pair every motion and condition file.

## Core layout

```text
DATA_ROOT/
├── motion_39d_30fps/<stem>.pt
├── condition_30fps/<stem>.pt
├── audio/<stem>.wav
├── transcripts/<stem>.txt
├── annotations/words/<stem>.json
├── raw_inputs.jsonl
├── splits/{train_drop,val_common}.txt
├── stats/drop_train.pt
├── audit/condition/per_clip_condition_motion_audit.csv
├── audit/condition/frozen_conditions.json
├── manifests/clips.jsonl
├── eval_assets/mmae/g1_mmae_30body_30fps.npy
└── assets/unitree_g1/
```

Extract all dataset and robot archives into the same root. The configs load the length CSV,
condition manifest and training statistics automatically. Keep these metadata files with the data.

## Physical motion

Load `.pt` files with `torch.load(path, map_location="cpu", weights_only=True)`.
Motion files contain `robot_repr[T,39]`, `real_num_frames`, `fps=30.0` and
`representation_schema="projecthermes-g1-39d-standard-v2"`.

| Slice | Meaning | Unit |
|---|---|---|
| `0:6` | Continuous 6D base orientation | Unitless |
| `6` | Frame-to-frame base yaw change | rad/frame |
| `7:10` | Yaw-local root velocity in XYZ order | m/s |
| `10:39` | 29 G1 joint angles in canonical order | rad |

Coordinates are Z-up with +X forward. Frame-zero yaw is removed from base orientation;
root velocity remains yaw-local. The evaluator/renderer reconstructs root motion and applies
the fixed joint mapping. These values are physical robot references, not XYZ joint positions
or MuJoCo qpos. Preserve their units and joint order.

## Frozen conditions

| Field | Shape or value |
|---|---|
| `audio_4fps` | `[Ta,1024]`; this historical key stores **30 FPS** features |
| `text_tokens` | `[N,2560]`; complete token sequence, `N <= 256` |
| `text_pooled` | `[2560]`, retained for compatibility |
| `token_times` | `[N,2]`; clip-relative start/end seconds |
| `n_tokens` | Actual tokenizer count, equal to `N` |
| `has_word_timing`, `latent_fps` | `True`, `30.0` |

New extracted conditions may use `audio_features` and `fps` aliases. Frozen audio/text
features and token intervals are stored as FP16; model inputs and token centers use FP32.
Text features use Qwen3.5-4B hidden layer −2. Audio features use Wav2Vec2 final hidden states
from mono 16 kHz input, linearly interpolated in FP32 to the target frame count with
`align_corners=True`, then stored in FP16. [Encoder identities](../manifests/condition_encoders.json)
and [download commands](INFERENCE.md#download-the-frozen-encoders) pin the upstream models.

### Complete-text validation

The loader accepts two verified condition formats:

- **Released frozen conditions:** the supplied `audit/condition/frozen_conditions.json`
  and its pinned SHA256 identify all 18,229 files, their text/time tensors and full token counts.
  The YAMLs already set both `data.condition_manifest` and `data.condition_manifest_sha256`;
  relative paths resolve against `data_root`.
- **New raw conditions:** `echo-g-extract-conditions` embeds `text_provenance` with schema
  `echo-g-complete-text-provenance-v1`, tokenizer count with `truncation=False`, complete token
  IDs/offsets, encoder identity, mapping rule and transcript/text/time digests. The loader
  validates those values against the stored transcript, words and tensors.

`echo-g-infer` requires explicit manifest arguments for released conditions; see
[cached inference](INFERENCE.md#cached-conditions-without-ground-truth-motion). An explicit
manifest always checks file identity, including files carrying embedded raw provenance.
Missing provenance or a mismatched identity is rejected. Regenerate incomplete raw features
with the extractor; editing counts or padding truncated features does not restore the transcript.

## Word and token times

`audio/<stem>.wav` preserves the segmented source waveform; `transcripts/<stem>.txt`
contains the source text. `annotations/words/<stem>.json` contains canonical text,
`words[text,start,end]`, token IDs and character offsets. Word times are relative to
that clip's audio start. Canonical tokenization text and original transcript bytes are
recorded separately.

Tokens can be subwords, spaces or punctuation. Canonical text joins word strings with spaces,
including non-English words. Each inserted space inherits the preceding word's end time;
a token interval is the union of its covered character times. Preserve zero-duration
intervals and the FP16 storage boundary before computing centers:

```text
frame_seconds[i] = i / 30
token_center[j] = mean(float32(token_times[j]))
signed_distance[i, j] = frame_seconds[i] - token_center[j]
```

The sign supports the learned lag; taking the absolute value changes attention behavior.

## Time and length

| Input path | Output length |
|---|---|
| Paired training, validation or sampling | Stored common audio/motion prefix, capped at 600 frames |
| Independent cached inference | Supplied length CSV, otherwise acoustic sequence length; maximum 600 |
| New audio | `max(2, round(audio_seconds * 30))`; reject audio over 20 seconds |
| New text only | Same rounding of the explicitly supplied duration; word times must fit it |

Text-bearing inputs require complete word-timed tokens, at most 256. Oversized raw inputs
are rejected without automatic truncation, segmentation or stitching. The learned position
table has capacity 608; the supported motion limit remains 600. Audio-only new requests do
not need transcripts. Output length is supplied to the model rather than predicted from text.

The CSV columns include `stem`, `aligned_frames` and `error`. `aligned_frames` fixes the common
valid prefix; errors or missing pairs are rejected. Use it for paired benchmarks: independently
rounding WAV duration can differ by one frame. The released text sequences contain at most
247 tokens (73 in validation).

## Normalization and evaluation

`stats/drop_train.pt` contains `[39]` mean/std tensors from the 14,987 training clips and
3,537,311 physical frames, using unbiased variance. Training normalizes motion; inference
reverses the transform using checkpoint-local statistics. Keep GT and predictions in physical
units, and use the published statistics rather than recomputing on another split.

The separate `eval_assets/mmae/g1_mmae_30body_30fps.npy` is the fixed BA normalization;
its population and aggregation are documented in [BENCHMARK.md](BENCHMARK.md).

## Raw requests

`raw_inputs.jsonl` uses one JSON object per clip. Audio paths are relative to the manifest
unless absolute. Stems start with a letter or number and otherwise contain letters, numbers,
`.`, `_` or `-`. Word times must be finite, with ordered start/end values. Use a new condition
directory when changing inputs or encoders. [INFERENCE.md](INFERENCE.md#new-audio-and-word-timed-text)
provides request examples and extraction commands for each model.
