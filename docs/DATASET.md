# Download and use the ECHO-G dataset

The dataset contains **18,229 clips**: 14,987 training and 3,242 validation. Each clip has
physical G1 motion, frozen audio/text conditions, segmented audio, transcript and word/token
annotations. Splits, training statistics and BA normalization are included.

The dataset is available at [gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G)
as a **private preview**. Only authorized accounts can download it. Public release and the
license for new contributions are pending. See the [data card](DATASET_CARD.md) for the
source, population and distribution terms.

## Download and extract

Install the standard Hugging Face CLI with `python -m pip install huggingface_hub` if needed.
Log in with an account authorized to access the private repository, then download the pinned
revision and extract the archives:

```bash
hf auth login
hf download gaopusen/ECHO-G --repo-type dataset \
  --revision 6abf2ddc129de54e2d950775023492f34fc71ac5 --local-dir data/echo-g

for archive in data/echo-g/data/*.tar; do
  tar -xf "$archive" -C data/echo-g
done
```

Extract all 12 archives into the same directory as the downloaded metadata. The resulting
`data/echo-g` is ready to use as `--data-root`; no preprocessing is needed for the supplied
frozen conditions. The total download is approximately **18.61 GB**, including 12 tar archives
(18.52 GB). Extracted clip files require another 18.45 GB while the archives are retained.

## Train or generate motion

After [installing ECHO-G](../README.md#installation), train with:

```bash
echo-g-train \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir outputs/audio_text \
  --device cuda
```

Model checkpoint upload is still pending. Once the checkpoint is available, generate the
validation split with:

```bash
echo-g-sample \
  --checkpoint weights/best.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir outputs/validation \
  --split val --seeds 0 --device cuda
```

The configurations read the supplied split lists, statistics and condition metadata automatically.
For tensor fields and timestamps, see [DATA_FORMAT.md](DATA_FORMAT.md). To evaluate or render
predictions, follow [BENCHMARK.md](BENCHMARK.md) and [VISUALIZATION.md](VISUALIZATION.md).
