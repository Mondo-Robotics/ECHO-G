# ECHO-G dataset and inference format

The ECHO-G release uses stable clip stems to pair conditions and physical robot references.
The split lists, not directory enumeration, determine the training and validation population.
The data card describes the exact [ECHO-G split and its source lineage](DATASET_CARD.md).

## Core layout

```text
DATA_ROOT/
├── motion_39d_30fps/<stem>.pt
├── condition_30fps/<stem>.pt
├── audio/<stem>.wav
├── transcripts/<stem>.txt
├── annotations/words/<stem>.json
├── raw_inputs.jsonl
├── eval_assets/mmae/g1_mmae_30body_30fps.npy
├── assets/unitree_g1/g1_mocap_29dof.xml
├── assets/unitree_g1/meshes/
├── assets/unitree_g1/LICENSE
├── splits/train_drop.txt
├── splits/val_common.txt
├── stats/drop_train.pt
├── audit/condition/per_clip_condition_motion_audit.csv
└── audit/condition/frozen_conditions.json
```

The supplied ECHO-G configurations read the length CSV and condition manifest automatically.
Both files are included with the dataset. See [download instructions](DATASET.md).

### Audio, transcripts and word times

`audio/<stem>.wav` preserves the packaged waveform and its sample rate without further cropping
or stretching. `transcripts/<stem>.txt` preserves the original packaged text bytes. The paired
`annotations/words/<stem>.json` records `canonical_transcript`, `words` entries with `text`,
`start`, and `end`, and `tokenization` metadata including token IDs and character offsets.
Word times are in seconds relative to the start of that clip's WAV. The original transcript and
canonical tokenization text are distinct identities even when their visible wording matches.
The word-to-token mapping follows the frozen ECHO-G convention below.

`raw_inputs.jsonl` supplies the corresponding audio path, transcript and words to
`echo-g-extract-conditions`. Use the released canonical lengths for cached benchmark inference;
the untrimmed waveform duration is not a replacement for the stored common-prefix length.
The benchmark reads `--wav-dir DATA_ROOT/audio` and uses the frozen MMAE array above.

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

Upstream encoder repository IDs, fixed revisions and download commands are in
[the encoder download guide](../README.md#download-the-frozen-encoders);
[the encoder manifest](../manifests/condition_encoders.json) records the runtime file hashes.

### Complete-text provenance

`n_tokens == len(text_tokens)` alone does not prove that the full transcript was encoded.
The loader requires one of the following verified paths.

**New raw conditions** include a `text_provenance` mapping with schema
`echo-g-complete-text-provenance-v1`, produced automatically by `echo-g-extract-conditions`:

| Field | Purpose |
|---|---|
| `full_token_count` | Count from the tokenizer with `truncation=False`, independent of the stored feature-row count |
| `tokenizer_truncation` | Must be `False` |
| `token_ids`, `token_offsets` | Complete token IDs and character offsets, each with `full_token_count` entries |
| `encoder_identity` | Model identifier, encoder/tokenizer classes, and resolved revision when supplied by the encoder |
| `token_time_mapping` | `character_offset_union_with_spaces_at_previous_word_end` |
| `canonical_transcript_sha256` | Binds the exact canonical transcript |
| `text_tokens_sha256`, `token_times_sha256` | Bind the stored text features and token intervals |

The condition payload retains `canonical_transcript`, `word_timestamps`, `source_text_model`,
`hidden_layer=-2`, and `encoder_dtype=bfloat16`. Loading recomputes word-derived token intervals
from the stored canonical transcript and offsets, checks the encoder metadata, and verifies
counts and tensor digests. These fields ensure that loaded tokens and intervals match the extraction inputs.

**Published frozen conditions** require the released condition manifest and its
externally pinned SHA256:

```text
Dataset path: audit/condition/frozen_conditions.json
SHA256: 81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01
Entries: 18229
```

The dataset manifest records each frozen condition file, its text/time tensors and complete
tokenizer count. The provided ECHO-G configurations already select this manifest.

Paired training, validation and sampling use `data.condition_manifest` and
`data.condition_manifest_sha256` from YAML. Relative manifest paths resolve against `data_root`;
absolute paths are also supported. Both fields must be supplied together. The provided ECHO-G YAMLs
pin the official path and SHA above. Independent `echo-g-infer` uses the explicit
`--condition-manifest PATH --condition-manifest-sha256 SHA256` options; it does not infer a data
root or silently reuse the YAML's relative manifest path.

When a manifest is explicitly supplied, file identity is checked even if the payload also
contains new raw provenance. Without either valid raw provenance or a matching released file identity,
the loader rejects the cache. Complete inputs of any length up to 256 tokens remain valid
through either verified path.

Do not pad truncated features, edit counts, or recompute a manifest from unverified caches to
make them pass. Regenerate raw conditions with the current extractor, or use the official
frozen data and its published manifest identity. Changing frozen condition files invalidates the supplied manifest.

## Time and length

For timed inputs, ECHO-G constructs:

```text
frame_seconds[i] = i / 30
token_center[j] = mean(float32(token_times[j]))
signed_distance[i, j] = frame_seconds[i] - token_center[j]
```

Keep the sign. An absolute value changes the learned-lag behavior. Tokens are subwords/tokenizer
units, not words; their intervals come from the original word-to-character-to-token mapping.
The raw extractor joins word strings with spaces, including non-English words, and assigns each
inserted space the preceding word's end time, preserving the reference convention. ECHO-G does not
replace this mapping with word pooling or a new alignment algorithm.

| Entry point | Frame policy | Text policy |
|---|---|---|
| Paired training/validation Dataset | Stored common audio/motion prefix, at most 600 frames | Complete stored tokens, at most 256; require verified provenance or manifest identity |
| Independent cached inference | Acoustic sequence length, or supplied canonical length; at most 600 | Reject oversized conditions or missing/invalid complete-text provenance |
| Raw audio extraction → inference | `T = max(2, round(audio_seconds * 30))`; reject audio longer than 20 seconds | Require word timing; reject more than 256 tokens |

There is no automatic segmentation or stitching. For longer inputs, create shorter audio clips
and corresponding clip-relative transcript times before inference. Silence is included in the
audio duration. Learned positional capacity is 608 internally; it does not extend the supported
600-frame input limit. The model receives the output length externally rather than predicting an
end-of-utterance token.

The condition decoder requires finite, word-derived token intervals and
`has_word_timing=True`. Missing or invalid timing is rejected.

## Training statistics

```python
{
    "mean": torch.Tensor,  # [39]
    "std": torch.Tensor,   # [39], positive
}
```

Use only `stats/drop_train.pt` for the reference ECHO-G run. It was computed from the 14,987 training
clips (3,537,311 physical frames) with unbiased variance, without validation data. Training
normalizes motion and inference reverses the same transform using checkpoint-local statistics.
Do not apply normalization to the ground-truth files themselves or substitute statistics
computed from another population.

## Canonical frame lengths

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
