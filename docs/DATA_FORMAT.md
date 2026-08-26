# Dataset Format

ECHO-G trains from paired condition and physical robot-motion files. Every file is named
`<stem>.pt`, and split files contain one stem per line. Missing pairs and duplicate split entries
are treated as errors.

## Motion Payload

```python
{
    "robot_repr": torch.Tensor,       # [T, D], float, unnormalized physical values
    "real_num_frames": int,           # valid prefix length
    "fps": 30.0,
    "representation_schema": str,
}
```

The paper dataset uses `D=39`:

| Slice | Meaning | Unit |
|---|---|---|
| `[0:6]` | continuous 6D base orientation | unitless |
| `[6]` | frame-to-frame base yaw change | rad |
| `[7:10]` | yaw-local root velocity in standard XYZ order | m/s |
| `[10:39]` | robot joint positions | rad |

The coordinate system is Z-up, the forward axis is +X, and frame-zero yaw is removed from the
base orientation. The joint-position order must match the downstream robot tracker.

## Condition Payload

```python
{
    "audio_features": torch.Tensor,   # [Ta, A], frame-aligned acoustic hidden states
    "text_tokens": torch.Tensor,      # [N, L], transcript-token hidden states
    "text_pooled": torch.Tensor,      # [L], fallback when token features are absent
    "token_times": torch.Tensor,      # [N, 2], token start/end times in seconds
    "has_word_timing": bool,
    "fps": 30.0,
}
```

The frozen paper release uses `A=1024` and `L=2560`. For compatibility with the original research
release, the data loader also accepts `audio_4fps` as the acoustic-feature key; its actual rate is
read from the dataset contract and is 30 FPS for the paper experiments.

When timing is unavailable, set `has_word_timing` to `false`. SGDiT then performs ordinary token
cross-attention because the time-distance matrix is zero.

## Statistics

The statistics file contains train-split-only values:

```python
{
    "mean": torch.Tensor,  # [D]
    "std": torch.Tensor,   # [D], strictly positive
}
```

Training normalizes motion as `(robot_repr - mean) / std`. Sampling reverses that transform before
writing predictions. Ground-truth files must remain in physical units.

## Alignment Audit

The configured CSV must contain at least:

```csv
stem,aligned_frames,error
example_0001,312,
```

`aligned_frames` freezes the common valid prefix of motion and acoustic features. Any nonempty
`error` field aborts loading. This makes temporal alignment part of the versioned dataset rather
than an implicit runtime heuristic.

## Condition Manifest

Raw condition extraction reads JSON Lines:

```json
{"stem":"example_0001","audio_path":"audio/example.wav","transcript":"Hello world","words":[{"text":"Hello","start":0.10,"end":0.42},{"text":"world","start":0.55,"end":0.91}]}
```

`audio_path` may be absolute or relative to the manifest. A word can optionally provide
`char_start` and `char_end`; otherwise the extractor aligns word strings to the transcript in
order. Stems may contain only letters, numbers, `.`, `_`, and `-`.
