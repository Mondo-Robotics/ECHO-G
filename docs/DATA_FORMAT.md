# V2 dataset and inference format

The V2 release uses stable clip stems to pair conditions and physical robot references.
The split lists, not directory enumeration, determine the training and validation population.
The data card describes the exact [V2 split and its source lineage](DATASET_CARD.md).

## Core layout

```text
DATA_ROOT/
├── motion_39d_30fps/<stem>.pt
├── condition_30fps/<stem>.pt
├── splits/train_drop.txt
├── splits/val_common.txt
├── stats/drop_train.pt
└── audit/condition/per_clip_condition_motion_audit.csv
```

The Hugging Face package will additionally contain word-time annotations and provenance/checksum
manifests. Their packaged filenames and shard hashes are pending; they are not required by the
cached-condition model loader. Raw transcript text and source media have separate distribution terms.

## Physical motion

```python
{
    "robot_repr": torch.Tensor,       # [T, 39], unnormalized physical values
    "real_num_frames": int,           # valid prefix length
    "fps": 30.0,
    "representation_schema": "projecthermes-g1-39d-standard-v2",
}
```

| Slice | Meaning | Unit |
|---|---|---|
| `[0:6]` | Continuous 6D base orientation | Unitless |
| `[6]` | Frame-to-frame base yaw change | Radians per frame |
| `[7:10]` | Yaw-local root velocity, standard XYZ order | m/s |
| `[10:39]` | 29 G1 joint angles in the canonical representation order | Radians |

Coordinates are Z-up and forward is +X. Only the base orientation has frame-zero yaw removed.
Root velocity remains yaw-local. The representation is not an array of XYZ body positions or
MuJoCo qpos; the evaluator/renderer reconstructs root motion and applies the fixed G1 joint mapping.
Preserve the canonical joint order rather than inferring it from URDF or filesystem ordering.
Ground-truth and exported prediction files remain in physical units.

## Frozen conditions

```python
{
    "audio_4fps": torch.Tensor,       # [Ta, 1024], historical key; ACTUALLY 30 FPS
    "text_tokens": torch.Tensor,     # [N, 2560], complete token sequence, N <= 256
    "text_pooled": torch.Tensor,     # [2560], retained for payload compatibility
    "token_times": torch.Tensor,     # [N, 2], clip-relative start/end in seconds
    "has_word_timing": True,
    "n_tokens": int,                 # actual tokenizer count, equal to N
    "latent_fps": 30.0,
}
```

The package also accepts `audio_features` and `fps` for newly extracted conditions. These are
aliases for the same time-aligned audio contract. Frozen audio/text features and token intervals
cross an FP16 storage boundary; model inputs and token-center calculations use FP32.

Text is produced by Qwen3.5-4B hidden layer −2. Acoustic features use Wav2Vec2 final hidden states:
16 kHz mono input, native encoder time sequence, FP32 linear interpolation to the target frame
count with `align_corners=True`, then FP16 storage. The audio key name does not denote a 4 Hz rate.

Full token features and an integer `n_tokens` equal to the stored token count are required for
V2. A historical 64-token cache is invalid if the transcript
actually contains more than 64 tokens. Do not pad such a cache to 256 or change its metadata to
make it appear complete. Historical frozen caches can lack new encoder/source identity fields, but must retain the
required count and timing fields. Release identity and audits establish their provenance.

## Time and length

For timed inputs, V2 constructs:

```text
frame_seconds[i] = i / 30
token_center[j] = mean(float32(token_times[j]))
signed_distance[i, j] = frame_seconds[i] - token_center[j]
```

Keep the sign. An absolute value changes the learned-lag behavior. Tokens are subwords/tokenizer
units, not words; their intervals come from the original word-to-character-to-token mapping.
The raw extractor joins word strings with spaces, including non-English words, and assigns each
inserted space the preceding word's end time, preserving the reference convention. V2 does not
replace this mapping with word pooling or a new alignment algorithm.

| Entry point | Frame policy | Text policy |
|---|---|---|
| Paired training/validation Dataset | Audited common audio/motion prefix, at most 600 frames | Complete stored tokens, at most 256; reject known truncated caches |
| Independent cached inference | Acoustic sequence length, or supplied canonical audited length; at most 600 | Reject oversized or known truncated conditions |
| Raw audio extraction → inference | `T = max(2, round(audio_seconds * 30))`; reject audio longer than 20 seconds | Require word timing; reject more than 256 tokens |

There is no automatic segmentation or stitching. For longer inputs, create shorter audio clips
and corresponding clip-relative transcript times before inference. Silence is included in the
audio duration. Learned positional capacity is 608 internally; it does not extend the supported
600-frame input limit. The model receives the output length externally rather than predicting an
end-of-utterance token.

The original training implementation permitted missing/invalid timing by passing zero distances,
which makes local and global attention mathematically coincide. The V2 condition decoder in this
release requires finite, word-derived token intervals and `has_word_timing=True`; it does not
silently replace an invalid public input with that historical fallback.

## Training statistics

```python
{
    "mean": torch.Tensor,  # [39]
    "std": torch.Tensor,   # [39], positive
}
```

Use only `stats/drop_train.pt` for the reference V2 run. It was computed from the 14,987 training
clips (3,537,311 physical frames) with unbiased variance, without validation data. Training
normalizes motion and inference reverses the same transform using checkpoint-local statistics.
Do not apply normalization to the ground-truth files themselves. The all-status training
statistics belong to another experiment and must not be substituted.

## Alignment audit

The CSV includes at least:

```csv
stem,aligned_frames,error
example_0001,312,
```

`aligned_frames` freezes the common valid prefix of audio and motion. A nonempty error or missing
required pair is an error. Benchmark inference must preserve these canonical lengths; using raw
WAV rounding independently can differ by one frame on the released dataset.

## Raw-input manifest

Use one JSON object per line:

```json
{"stem":"example_0001","audio_path":"audio/example.wav","transcript":"Hello world","words":[{"text":"Hello","start":0.10,"end":0.42},{"text":"world","start":0.55,"end":0.91}]}
```

Paths are relative to the manifest unless absolute. Stems start with a letter or number and otherwise use letters, numbers, `.`, `_`, and `-`.
Word start/end times are in seconds relative to the supplied audio, finite, and ordered within
each interval. Use a new condition directory when changing source audio/text or encoder assets.
The complete three-stage extraction and sampling commands are in the [README](../README.md).
