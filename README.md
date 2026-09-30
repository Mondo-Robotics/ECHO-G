# ECHO-G

**Embodied Co-speech Humanoid mOtion Generation**

ECHO-G generates full-body Unitree G1 motion from audio and a word-timed transcript.
It generates physical 39D robot references at 30 FPS using a rectified-flow transformer.
ECHO-G combines normalized Q/K content attention with global/local text attention and a learned,
directional Gaussian time prior.

```text
Audio -> frozen Wav2Vec2 -> 30 FPS acoustic features ---+
                                                      v
Gaussian motion noise -> motion + audio + positions -> ECHO-G DiT -> physical robot39
                                                      ^
Word-timed text -> frozen Qwen -> token features -------+
                       global / local Gaussian cross-attention
```

This package provides the model, training and inference commands, a benchmark evaluator, and
MuJoCo visualization for the released robot-motion dataset.

## Data and weights

**The dataset and model weights are available as private previews on Hugging Face.**
Access requires an account authorized for each repository; public release remains pending.
ECHO-G data contributions and project model/evaluation weights use **CC BY-NC 4.0**;
third-party materials retain their applicable terms. See [licensing scope](docs/LICENSING.md).

| Asset | Hugging Face location | Status |
|---|---|---|
| Processed robot-motion dataset | [gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G) | Uploaded private preview |
| ECHO-G audio+text `best.pt` (15k, EMA) | [Model repository](https://huggingface.co/gaopusen/ECHO-G/tree/2026-09-30) | Uploaded private preview |
| Benchmark FGD encoder | [Model repository: evaluation](https://huggingface.co/gaopusen/ECHO-G/tree/2026-09-30/evaluation) | Uploaded private preview |
| Frozen BA normalization | With dataset | Included in the private dataset preview |
| G1 MuJoCo XML and meshes | With dataset | Included under BSD-3-Clause |

The prepared dataset contains processed robot motions, frozen audio/text conditions, clip-aligned
audio, source transcripts, word-time annotations, frozen splits, training statistics, and the
frozen BA normalization array. The dataset split is **14,987 training clips + 3,242 validation clips**.
Audio is stored as `audio/<stem>.wav`, source text as `transcripts/<stem>.txt`, and canonical text,
word times, and tokenizer mapping as `annotations/words/<stem>.json`. `raw_inputs.jsonl` provides
inputs for feature extraction. BEAT2-derived audio/text retain their applicable source license
and attribution, with clipping/text reconstruction documented. Encoder weights are separate downloads with their own terms. The dataset includes the G1
MuJoCo XML and meshes with their BSD-3-Clause license; see [visualization](docs/VISUALIZATION.md).

The public benchmark reports FGD, Div, BA, weighted jerk, foot metrics, and optional MM20. It does
not require or distribute semantic labels. See [the benchmark protocol](docs/BENCHMARK.md).

- [ECHO-G model card](docs/MODEL_CARD.md)
- [Dataset card](docs/DATASET_CARD.md)
- [Download and use the dataset](docs/DATASET.md)
- [Data format and timing contract](docs/DATA_FORMAT.md)
- [Dataset asset manifest](manifests/dataset.json)
- [Frozen-condition manifest identity](manifests/frozen_conditions.json)
- [Model asset manifest](manifests/audio_text.json)
- [Benchmark asset manifest](manifests/benchmark_assets.json)

Follow the [dataset download guide](docs/DATASET.md), then use the commands below.
The clip data comprise 12 tar archives (18.52 GB), with separate metadata and a robot-asset archive.
Download and extract all archives into the same data root. The dataset revision is
`2026-09-29-license`; the model revision is `2026-09-30`.

### Download model weights

Install the Hugging Face CLI with `python -m pip install huggingface_hub` if needed.
Log in with an account authorized for the private model repository, then download and verify:

```bash
hf auth login
hf download gaopusen/ECHO-G --repo-type model \
  --revision 2026-09-30 --local-dir weights
(cd weights && sha256sum -c SHA256SUMS)
```

This supplies `weights/best.pt`, `weights/evaluation/g1_fgd_encoder.pt`,
`weights/configs/sgdit_audio_text.yaml`, the model card, license and checksums.
The model and dataset are separate repository types with independent pinned revisions.
No human-motion VAE is needed for this direct robot model. See the
[model card](docs/MODEL_CARD.md) for file identities and release checks.

## Installation

Use Python 3.10 or 3.11 and a compatible PyTorch installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For extracting conditions from raw input:

```bash
python -m pip install -e ".[preprocess]"
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Benchmark and MuJoCo dependencies are documented in [BENCHMARK.md](docs/BENCHMARK.md) and
[VISUALIZATION.md](docs/VISUALIZATION.md).

## Model at a glance

| Setting | ECHO-G audio+text |
|---|---|
| Backbone | 12 blocks, width 768, 8 heads, FFN 2048 |
| Motion representation | Direct physical 39D G1, 30 FPS |
| Position encoding | Learned; model capacity 608, valid input limit 600 frames |
| Text | Qwen3.5-4B layer −2, 2,560 dimensions; at most 256 tokens |
| Audio | Wav2Vec2 final hidden state, 1,024 dimensions; interpolated to motion frames |
| Cross-attention | QK normalization; global/local token branches sharing Q/K/V |
| Time prior | Learned per-head width, signed lag, and mixture; support-gated Gaussian |
| Sampling | EMA, 8 Euler updates, CFG = 1 |
| Reference weights | Minimum-EMA-validation-loss `best.pt`, step 15,000 |

The reference checkpoint was selected at step 15,000 by minimum EMA validation loss from
63,000 training steps. ECHO-G uses `frame_seconds - token_center_seconds`, preserving the sign needed for
learned lag. Text keys remain tokenizer units, including subwords; ECHO-G does not pool them into
one key per word. See [the architecture](docs/ARCHITECTURE.md).

## Training

Expected core layout:

```text
DATA_ROOT/
├── condition_30fps/<stem>.pt
├── motion_39d_30fps/<stem>.pt
├── splits/train_drop.txt
├── splits/val_common.txt
├── stats/drop_train.pt
├── audit/condition/per_clip_condition_motion_audit.csv
└── audit/condition/frozen_conditions.json
```

The supplied ECHO-G configurations use these files automatically. Download and extract the complete
dataset, then run:

```bash
echo-g-train \
  --config configs/sgdit_audio_text.yaml \
  --data-root /path/to/echo-g-data \
  --output-dir outputs/audio_text \
  --device cuda
```

Resume using the same command plus `--resume outputs/audio_text/latest.pt`.
The training directory records `experiment.json`, `latest.pt`, `best.pt`, `final.pt`, and
`complete.json`. `latest.pt` is the full training-resume state; `best.pt` is selected by validation
loss, not by FGD. The training recipe is in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Inference

### Frozen validation split

Use the paired dataset entry point when reproducing the canonical benchmark:

```bash
echo-g-sample \
  --checkpoint weights/best.pt \
  --config weights/configs/sgdit_audio_text.yaml \
  --data-root /path/to/echo-g-data \
  --output-dir results/audio_text \
  --split val --seeds 0 --device cuda
```

### Cached conditions without ground-truth motion

For new conditions produced by the current raw extractor, use the embedded complete-text
provenance:

```bash
echo-g-infer \
  --checkpoint weights/best.pt \
  --config weights/configs/sgdit_audio_text.yaml \
  --condition-dir /path/to/condition_30fps \
  --output-dir outputs/new_clips \
  --device cuda
```

For independent inference on released dataset conditions, supply the included condition manifest:

```bash
echo-g-infer \
  --checkpoint weights/best.pt \
  --config weights/configs/sgdit_audio_text.yaml \
  --condition-dir /path/to/echo-g-data/condition_30fps \
  --condition-manifest /path/to/echo-g-data/audit/condition/frozen_conditions.json \
  --condition-manifest-sha256 81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01 \
  --output-dir outputs/frozen_conditions \
  --device cuda
```

Both manifest options are required together. Independent inference does not infer a dataset
root from the YAML. An explicitly supplied manifest always verifies condition-file identity,
even if a file also has raw provenance.

Use `--stem-list /path/to/stems.txt` to select clips. For canonical lengths, also provide
`--lengths-csv /path/to/per_clip_condition_motion_audit.csv`. Otherwise the cached acoustic
sequence supplies the output frame count. Predictions are denormalized physical robot references.

### Raw audio and word-timed text

The dataset includes `raw_inputs.jsonl` for its released clips. For your own inputs, create
`utterances.jsonl` with paths relative to that manifest:

```json
{"stem":"example_0001","audio_path":"audio/example.wav","transcript":"Hello world","words":[{"text":"Hello","start":0.10,"end":0.42},{"text":"world","start":0.55,"end":0.91}]}
```

Word times are relative to the audio clip. The ECHO-G extractor reconstructs the canonical text
from the word list and preserves the original token/time mapping.

#### Download the frozen encoders

Run the following commands from the ECHO-G checkout. These are separate upstream model
repositories, not files in the ECHO-G model/dataset repository. Their upstream licenses apply.
Encoder downloads are only needed for **new raw inputs**; inference on the released frozen
conditions does not require either model.

| Encoder | Upstream Hugging Face repository | Fixed revision (full commit SHA) |
|---|---|---|
| Text | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Audio | [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/tree/569a6236e92bd5f7652a0420bfe9bb94c5664080) | `569a6236e92bd5f7652a0420bfe9bb94c5664080` |

These revisions match the encoder file identities recorded during the raw-input acceptance.
The Wav2Vec2 weights and configuration match the copy used from the
`wav2vec2-large-xlsr-53-english` subdirectory of `Wan-AI/Wan2.2-S2V-14B`;
the standalone repository avoids downloading the Wan video model. The exact identities,
loader classes and SHA256 values are recorded in
[the encoder manifest](manifests/condition_encoders.json).

```bash
python -m pip install -e ".[preprocess]"
python -m pip install huggingface_hub

hf download Qwen/Qwen3.5-4B \
  --repo-type model \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B

hf download jonatasgrosman/wav2vec2-large-xlsr-53-english \
  config.json preprocessor_config.json special_tokens_map.json vocab.json \
  model.safetensors README.md \
  --repo-type model \
  --revision 569a6236e92bd5f7652a0420bfe9bb94c5664080 \
  --local-dir models/wav2vec2-large-xlsr-53-english

sha256sum -c manifests/condition_encoders.sha256
```

Allow roughly **10.6 GB** for the model files. The audio command intentionally downloads
only the safetensors weights and processor files, not duplicate PyTorch/Flax weights or the
ASR language model. Check that every listed file reports `OK` before extracting conditions.
Use clean, dedicated model directories; do not mix files from different revisions.
The extractor takes these **local directories**, not an unpinned Hub ID. Keep the
`--revision` values above unchanged when preparing another machine.

#### Extract and infer

Using the verified downloads, run:

```bash
echo-g-extract-conditions \
  --mode audio --manifest utterances.jsonl \
  --output-dir data/new_conditions \
  --audio-model models/wav2vec2-large-xlsr-53-english \
  --fps 30 --device cuda

echo-g-extract-conditions \
  --mode text --manifest utterances.jsonl \
  --output-dir data/new_conditions \
  --text-model models/Qwen3.5-4B \
  --text-layer -2 --max-text-tokens 256 --device cuda

echo-g-extract-conditions \
  --mode merge --manifest utterances.jsonl \
  --output-dir data/new_conditions

echo-g-infer \
  --checkpoint weights/best.pt \
  --config weights/configs/sgdit_audio_text.yaml \
  --condition-dir data/new_conditions \
  --output-dir outputs/raw \
  --device cuda
```

The extractor writes `echo-g-complete-text-provenance-v1` metadata: the full tokenizer count,
complete IDs/offsets, encoder and word-time provenance, and text/time tensor digests. These new
conditions can be loaded without the frozen-dataset manifest.

Output duration comes from the audio, with `T = max(2, round(duration_seconds * 30))`.
Raw clips over 20 seconds or 256 text tokens are rejected; the pipeline does not automatically
split, truncate, or stitch them. See [the timing and length rules](docs/DATA_FORMAT.md#time-and-length).

Use frozen conditions for historical benchmark reproduction. Fresh encoder outputs can vary
with encoder revisions, decoding, hardware, and backend; exact agreement with the historical
audio or text cache is not claimed.

## Benchmark and visualization

- Run the integrated latest `scripts/eval_g1_motion_cls.py` following
  [BENCHMARK.md](docs/BENCHMARK.md). Required evaluator assets are tracked separately from weights.
- Render physical robot references using the MuJoCo pipeline in
  [VISUALIZATION.md](docs/VISUALIZATION.md).

The reference checkpoint scores on the 3,242-clip validation split were **FGD 2.278349** (seed000) and
**MM20 1.785556** (seeds 0–19). The packaged implementation scores **FGD 2.278311** under the released benchmark protocol;
MM20 refers to the original 20-seed run.
[Reproduction scope and reference results](docs/REPRODUCIBILITY.md#reference-results)
explain the distinction.

## Project layout

```text
configs/                  training and inference configuration
src/echo_g/               model, data, training, inference, and condition extraction
scripts/                  G1 evaluation and MuJoCo visualization entry points
docs/                     architecture, data/model cards, and usage protocols
manifests/                asset identities and Hugging Face distribution status
tests/                    implementation and integration checks
```

## License and citation

Code retains **PolyForm Noncommercial 1.0.0** under [LICENSE](LICENSE) and [NOTICE](NOTICE).
ECHO-G data contributions and project model/evaluation weights use **CC BY-NC 4.0** under
[LICENSE-DATA-WEIGHTS](LICENSE-DATA-WEIGHTS). Third-party materials retain their applicable
licenses. See [licensing scope and contacts](docs/LICENSING.md). Citation metadata is in
[CITATION.cff](CITATION.cff).
