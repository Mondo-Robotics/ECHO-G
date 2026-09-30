# Download and use the ECHO-G dataset

The dataset contains **18,229 clips**: 14,987 training and 3,242 validation. Each clip has
physical G1 motion, frozen audio/text conditions, segmented audio, transcript and word/token
annotations. Splits, training statistics and BA normalization are included.

The dataset is available at [gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G)
as a **public release**. Downloads do not require access approval. ECHO-G data contributions use **CC BY-NC 4.0**; third-party materials retain their
applicable terms. See the [data card](DATASET_CARD.md) and [licensing scope](LICENSING.md).

## Download and extract

Install the standard Hugging Face CLI with `python -m pip install huggingface_hub` if needed.
Download the pinned revision and extract the archives:

```bash
hf download gaopusen/ECHO-G --repo-type dataset \
  --revision v0.2.0 --local-dir data/echo-g

for archive in data/echo-g/data/*.tar data/echo-g/robot_assets/*.tar; do
  tar -xf "$archive" -C data/echo-g
done
```

Extract all 12 clip-data archives and the robot-asset archive into the same directory as the
downloaded metadata. The resulting `data/echo-g` is ready to use as `--data-root`; no preprocessing
is needed for the supplied frozen conditions. The clip-data archives occupy 18.52 GB; extracted
clip files require another 18.45 GB while those archives are retained. The robot-asset archive
adds about 19.73 MB.

The rendering model is then available at
`data/echo-g/assets/unitree_g1/g1_mocap_29dof.xml`, with its referenced meshes and license.

## Train or generate motion

After [installing ECHO-G](../README.md#installation), train with:

```bash
echo-g-train \
  --config configs/sgdit_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir outputs/audio_text \
  --device cuda
```

Download the public model repository at revision `v0.2.0` using the
[weight download commands](../README.md#download-model-weights), then generate the validation
split with:

```bash
echo-g-sample \
  --checkpoint weights/best.pt \
  --config weights/configs/sgdit_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir results/audio_text \
  --split val --seeds 0 --device cuda
```

The configurations read the supplied split lists, statistics and condition metadata automatically.
For tensor fields and timestamps, see [DATA_FORMAT.md](DATA_FORMAT.md). To evaluate or render
predictions, follow [BENCHMARK.md](BENCHMARK.md) and [VISUALIZATION.md](VISUALIZATION.md).
