# Paper Reproduction Protocol

## Frozen Setup

- Robot representation: physical 39D reference
- Frame rate: 30 FPS
- Maximum training duration: 600 frames (20 seconds)
- Position encoding: sinusoidal
- Motion generator: 12 transformer blocks, width 768, 8 heads
- Flow solver at inference: 8 Euler steps
- Random seed: 20260825
- Precision: FP32, AMP and TF32 disabled
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-4`
- Gradient clipping: `1.0`
- EMA decay: `0.999`
- Batch size: 3
- Gradient accumulation: 16
- Effective batch size: 48
- Stop condition: 63,000 optimizer steps

These values are stored in `configs/sgdit_sinusoidal.yaml`.

## Frozen Dataset Identity

The paper run used 16,940 training utterances and 3,242 speaker-held-out validation utterances.
The separately distributed data release can be verified with these hashes:

| Artifact | SHA256 |
|---|---|
| release version | `1491083788e5dbbd5c19e737186f03d88ac585087548d6aa0a15ec69e0230941` |
| `splits/train_all.txt` | `18984dc04394a76675c6bca61d5d53ba8e4f77f21fd5a844ed0b60ac8f254035` |
| `splits/val_common.txt` | `36a960134a321f7b2c47c1b24aabfdf1fa8ee213ea59a8570740be196ba7d83b` |
| `stats/all_train.pt` | `89b5c2dffadb8ddea7a206eea6b381a78878262e31ba5381a7d361441287972e` |

The data validator checks pairing, dimensions, finite values, statistics, and audited lengths. The
training command additionally records the split and statistics hashes in `experiment.json` and all
checkpoints.

## Reference Training Run

The reference direct audio-and-text sinusoidal run reached 63,000 optimizer steps. Its best EMA
checkpoint had validation flow objective `0.275736` and SHA256:

```text
d7ddaebaf834d5fa26a1edac5b11d7d933145269c0ca76c5f5dea3ecf09c8e95
```

Minor floating-point differences can occur across PyTorch, CUDA, and GPU versions. Compare
checkpoint metadata, validation trends, and downstream benchmark results rather than expecting
bitwise-identical model weights across hardware.

## Commands

```bash
echo-g-validate-data \
  --config configs/sgdit_sinusoidal.yaml \
  --data-root "$ECHO_G_DATA"

echo-g-train \
  --config configs/sgdit_sinusoidal.yaml \
  --data-root "$ECHO_G_DATA" \
  --output-dir outputs/sgdit_sinusoidal \
  --device cuda

echo-g-sample \
  --checkpoint outputs/sgdit_sinusoidal/best.pt \
  --data-root "$ECHO_G_DATA" \
  --output-dir outputs/sgdit_sinusoidal/predictions/val \
  --split val --seeds 0 --device cuda
```
