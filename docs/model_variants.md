# Model variants

All four models generate physical Robot39 motion at 30 FPS and use the same
[robot dataset](https://huggingface.co/datasets/gaopusen/ECHO-G),
`scripts/eval_g1_motion_cls.py`, and MuJoCo visualization pipeline.
The dataset and model repository remain private during release preparation.

| Model | Configuration | Weights | Training support |
|---|---|---|---|
| Audio + text | `configs/sgdit_audio_text.yaml` | `best.pt` | Included |
| Audio only | `configs/sgdit_audio_only.yaml` | `audio_only/best.pt` | Included |
| Text only | `configs/sgdit_text_only.yaml` | `text_only/best.pt` | Included |
| HumanRetarget | `configs/human_retarget_inference.yaml` | Three files in `human_retarget/` | Inference only |

HumanRetarget training code, training configurations, and human training targets
are outside this release. Its inference uses the shared speech conditions;
robot reference motions are needed only for evaluation. No human training dataset
or SMPL-X installation is required to run the released inference pipeline.

## Download

Install the package and authenticate with an account that has access to the
private repositories. The original audio + text revision `2026-09-30` remains
available. The revision below contains all four models.

```bash
hf download gaopusen/ECHO-G --revision 2026-09-30-models \
  --include 'audio_only/*' 'text_only/*' 'human_retarget/*' \
            'configs/*' 'evaluation/*' 'SHA256SUMS' 'LICENSE*' 'NOTICE' \
  --local-dir checkpoints/echo-g
```

Select only the variant directories you need. The FGD encoder in `evaluation/`
is shared. Model weights are licensed under CC BY-NC 4.0; source code remains
under PolyForm Noncommercial 1.0.0. See the root README for the separately pinned
dataset and upstream condition encoders.

## Inference on the released validation split

Set `DATA_ROOT` to the extracted robot dataset. Audio-only and text-only use
the same frozen conditions, splits, and robot normalization as audio + text.

```bash
echo-g-sample --config configs/sgdit_audio_only.yaml \
  --checkpoint checkpoints/echo-g/audio_only/best.pt \
  --data-root "$DATA_ROOT" --output-dir predictions/audio_only \
  --seeds 0 --steps 8 --guidance-scale 1 --batch-size 3 --device cuda

echo-g-sample --config configs/sgdit_text_only.yaml \
  --checkpoint checkpoints/echo-g/text_only/best.pt \
  --data-root "$DATA_ROOT" --output-dir predictions/text_only \
  --seeds 0 --steps 8 --guidance-scale 1 --batch-size 3 --device cuda

echo-g-sample --config configs/human_retarget_inference.yaml \
  --checkpoint checkpoints/echo-g/human_retarget/best.pt \
  --retarget-checkpoint checkpoints/echo-g/human_retarget/retarget_vae.pt \
  --human-vae-checkpoint checkpoints/echo-g/human_retarget/human_motion_vae.pt \
  --data-root "$DATA_ROOT" --output-dir predictions/human_retarget \
  --seeds 0 --steps 8 --guidance-scale 1 --batch-size 3 --device cuda
```

Outputs are stored in `seed_000/*.pt` with `robot_repr`, `real_num_frames`, FPS,
physical units, and input/weight identities. HumanRetarget exports the final
robot representation, ready for the common evaluator and renderer.

## Inputs and timing

Audio-only uses the original scaled-dot attention network with one zero text
key and no word-time constraint. Its text projection and cross-attention weights
remain active exactly as trained. It is not obtained by masking a different
audio + text checkpoint.

Text-only uses learned global/local word-time attention with zero acoustic
features. The full token sequence, up to 256 tokenizer units, and signed
`frame_time - token_center_time` distances are preserved. Tokens can be subwords,
spaces, or punctuation; word timestamps are mapped through character offsets.
The sign is necessary for the learned lag.

All models accept at most 600 motion frames. New audio is limited to 20 seconds;
longer inputs and text exceeding 256 tokens must be split explicitly. The loader
does not infer a duration from the number of words.

### Audio-only input without a transcript

The extraction manifest needs only an audio path and a safe clip name:

```json
{"stem":"example","audio_path":"example.wav"}
```

```bash
echo-g-extract-conditions --mode audio --manifest audio_requests.jsonl \
  --audio-model "$AUDIO_MODEL" --output-dir conditions --device cuda
echo-g-infer --config configs/sgdit_audio_only.yaml \
  --checkpoint checkpoints/echo-g/audio_only/best.pt \
  --condition-dir conditions/.audio --output-dir predictions/new_audio --device cuda
```

`AUDIO_MODEL` is the pinned wav2vec checkpoint described in the encoder download
instructions. Features are linearly interpolated to `max(2, round(duration * 30))`
frames with `align_corners=True`, matching the existing preprocessing path.

### Text-only input without audio

Supply an explicit duration in seconds and word times relative to the clip start:

```json
{"stem":"example","duration":2.0,"words":[{"text":"Hello","start":0.2,"end":0.6},{"text":"there.","start":0.7,"end":1.2}]}
```

```bash
echo-g-extract-conditions --mode text --manifest text_requests.jsonl \
  --text-model "$TEXT_MODEL" --output-dir conditions --device cuda
echo-g-infer --config configs/sgdit_text_only.yaml \
  --checkpoint checkpoints/echo-g/text_only/best.pt \
  --condition-dir conditions/.text --output-dir predictions/new_text --device cuda
```

`TEXT_MODEL` is the pinned Qwen checkpoint. Word times must fit the supplied
duration. Acoustic input is allocated as zeros; no audio encoder is run for this
path. When using released frozen conditions, provide the condition manifest and
its SHA256 as described in the existing inference instructions.

## HumanRetarget decoding

The generator predicts normalized Human136 motion. Inference restores its
physical units using the generator's statistics, normalizes it with the
RetargetVAE's human statistics, and decodes it to physical Robot39 motion.
The two sets of normalization statistics serve different stages.

The decoder loads both Human MotionVAE and the **complete** RetargetVAE state,
including the human encoder parameters updated for retargeting. It uses the
mean latent with windows of 120 frames, 20-frame chunks, and 20-frame overlap.
Short windows are padded and cropped back to their original lengths; the final
window preserves the tail. A +90-degree root orientation correction is applied
once. Do not repeat this correction before evaluation or visualization.

The HumanRetarget YAML is inference-only and is rejected by `echo-g-train`.
Both VAE files are checked against their published SHA256 before loading.

## Training the direct robot models

```bash
echo-g-train --config configs/sgdit_audio_only.yaml \
  --data-root "$DATA_ROOT" --output-dir runs/audio_only --device cuda
echo-g-train --config configs/sgdit_text_only.yaml \
  --data-root "$DATA_ROOT" --output-dir runs/text_only --device cuda
```

These configurations use the shared robot training split and robot statistics.
`best.pt` is selected by validation loss; it is not selected by the minimum FGD.
An independent training run need not reproduce the exact reported FGD.

## Reference scores and validation scope

The following are historical full-validation scores of the selected source
checkpoints, not new benchmark measurements from the release smoke tests.

| Model | Selected step | FGD ↓ | MM20 |
|---|---:|---:|---:|
| Audio only | 25,000 | 2.360242 | — |
| Text only | 15,000 | 2.436127 | 1.680639 |
| HumanRetarget, after robot decoding | 25,000 | 4.724520 | 1.498274 |

Weight cleanup preserves every network tensor and normalization statistic.
The port was checked against the preceding inference implementation on seven
real clips per added model with identical inputs, noise, precision, and device:
outputs were bitwise equal. HumanRetarget also matched exactly at 11 tested
lengths, including 61–100-frame sequences and 599/600 frames. These checks do
not promise bitwise agreement across different devices or PyTorch versions.


The fixed HF release `2026-09-30-models` resolves to
`8f2117c01e1b7d2a2e2df3bb75c3b513f03f644a`. All 19 supplied files were actually
downloaded and checked against their SHA256 values. Each of the four models then
passed condition-only inference, paired validation inference, and the common
benchmark pipeline on the same three real clips (840 frames per model). The
existing audio + text predictions remained bitwise equal. An additional real
247-token, 588-frame sample matched exactly for text-only and HumanRetarget.
These are release acceptance checks; the full-set scores above remain historical.

Each added model also passed MuJoCo export and video/audio checks on one 312-frame,
10.4-second clip using the released G1 assets. These videos validate the rendering
pipeline, not physical robot tracking.
