# Training

Training is provided for the three direct robot models: audio + text, audio-only and text-only.
HumanRetarget is inference-only; its human training data and training recipes are outside this release.

## Data and configuration

[Install ECHO-G](../README.md#installation), then download and extract the
[HF dataset](https://huggingface.co/datasets/gaopusen/ECHO-G#download) into `data/echo-g`.
The supplied YAMLs select the frozen conditions, common-prefix frame lengths, split lists and
training statistics automatically. No feature extraction is needed.

| Model | Configuration |
|---|---|
| Audio + text | `configs/sgdit_audio_text.yaml` |
| Audio only | `configs/sgdit_audio_only.yaml` |
| Text only | `configs/sgdit_text_only.yaml` |

```bash
echo-g-train --config configs/sgdit_audio_text.yaml \
  --data-root data/echo-g --output-dir outputs/audio_text --device cuda
```

Choose the matching config and a separate output directory for audio-only or text-only.
Keep `splits/train_drop.txt` (14,987 clips), `splits/val_common.txt` (3,242 clips) and
`stats/drop_train.pt` fixed. Statistics use only training motion: 3,537,311 physical frames.
The validation split holds out speakers `english_1_wayne_*`, `english_21_ayana_*` and
`english_3_solomon_*`.

## Resume and checkpoints

```bash
echo-g-train --config configs/sgdit_audio_text.yaml \
  --data-root data/echo-g --output-dir outputs/audio_text \
  --resume outputs/audio_text/latest.pt --device cuda
```

Resume from a training run's `latest.pt`. Published HF weights are inference packages without
optimizer-resume state. `best.pt` is selected by minimum EMA validation loss; FGD is measured
separately. The released audio+text checkpoint is EMA at step 15,000 from a 63,000-step run.
The released audio-only and text-only checkpoints are at steps 25,000 and 15,000 respectively.

## Reference recipe

The three direct-model configurations use:

| Setting | Value |
|---|---|
| Batch per GPU / gradient accumulation | 3 / 16 (effective batch 48 on one GPU) |
| Optimizer / learning rate / weight decay | AdamW / 0.0003 / 0.0001 |
| Schedule / optimizer steps | CosineAnnealingLR / 63,000; no warmup |
| Precision | FP32; AMP and TF32 disabled |
| Conditional dropout | 0.1, applied jointly to audio/text/time conditioning |
| Temporal loss weight | 0.5 |
| EMA / gradient clipping | 0.999 / 1.0 |
| Seed | 20260825 |
| Validation | Every 5,000 optimizer steps and at completion |
| Reference sampling | EMA, 8 Euler steps, CFG 1, seed 0 |

Validation during training uses batch size 3 and `drop_last=True`, so checkpoint selection
covers 3,240 clips. Full benchmark sampling covers all 3,242 clips and 891,351 aligned frames.
Architecture and attention parameters are described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Reproducing results

Use code tag `v0.2.0` and the HF revisions and hashes in [assets.json](../manifests/assets.json)
for the fixed release. Generate all validation clips and follow [BENCHMARK.md](BENCHMARK.md)
for reference scores and MM20.

Preserve complete token sequences (up to 256), signed frame-minus-token-center time differences,
canonical frame lengths, and the supplied normalization statistics. Token intervals cross an
FP16 storage boundary before FP32 center calculation; see [DATA_FORMAT.md](DATA_FORMAT.md).
Fresh feature extraction and independent training can differ numerically across PyTorch,
CUDA and GPU environments. Keep assets, configurations, seeds and evaluation settings fixed
when comparing runs.
