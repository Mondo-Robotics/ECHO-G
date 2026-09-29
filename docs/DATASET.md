# Download and use the V2 dataset

The dataset contains **18,229 clips**: 14,987 training and 3,242 validation. Each clip has
physical G1 motion, frozen audio/text conditions, segmented audio, transcript and word/token
annotations. Splits, training statistics and BA normalization are included.

The Hugging Face link and revision will be added after publication. See the
[data card](DATASET_CARD.md) for the source, population and distribution terms.

## Download and extract

Install the standard Hugging Face CLI with `python -m pip install huggingface_hub` if needed.
Replace the placeholders with the published dataset repository and revision:

```bash
hf download <HF_DATASET_REPO> --repo-type dataset \
  --revision <REV> --local-dir data/echo-g-v2

for archive in data/echo-g-v2/data/*.tar; do
  tar -xf "$archive" -C data/echo-g-v2
done
```

Extract all 12 archives into the same directory as the downloaded metadata. The resulting
`data/echo-g-v2` is ready to use as `--data-root`; no preprocessing is needed for the supplied
frozen conditions. The download is approximately 18.52 GB, plus 18.45 GB for extracted clip files.

## Train or generate motion

After [installing ECHO-G](../README.md#installation), train with:

```bash
echo-g-train \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root data/echo-g-v2 \
  --output-dir outputs/v2_audio_text \
  --device cuda
```

Or download the V2 checkpoint and generate the validation split:

```bash
echo-g-sample \
  --checkpoint weights/v2_best015000.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root data/echo-g-v2 \
  --output-dir outputs/v2_val \
  --split val --seeds 0 --device cuda
```

The configurations read the supplied split lists, statistics and condition metadata automatically.
For tensor fields and timestamps, see [DATA_FORMAT.md](DATA_FORMAT.md). To evaluate or render
predictions, follow [BENCHMARK.md](BENCHMARK.md) and [VISUALIZATION.md](VISUALIZATION.md).
