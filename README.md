# ECHO-G

**Embodied Co-speech Humanoid mOtion Generation**

ECHO-G generates full-body Unitree G1 motion at 30 FPS from speech and word-timed text.
The audio+text model uses a rectified-flow transformer with QK normalization,
global/local text attention, and a learned directional Gaussian time prior.

| Model | Configuration | Support |
|---|---|---|
| Audio + text | [sgdit_audio_text.yaml](configs/sgdit_audio_text.yaml) | Training and inference |
| Audio only | [sgdit_audio_only.yaml](configs/sgdit_audio_only.yaml) | Training and inference |
| Text only | [sgdit_text_only.yaml](configs/sgdit_text_only.yaml) | Training and inference |
| HumanRetarget | [human_retarget_inference.yaml](configs/human_retarget_inference.yaml) | Inference only |

All models output physical 39D robot motion. Inputs support at most **20 seconds / 600 frames**
and **256 text tokens**. HumanRetarget training data and training recipes are outside this release.

## Installation

Use Python 3.10 or 3.11 and a compatible PyTorch installation. Run the commands from this checkout:

```bash
git clone --branch v0.2.0 https://github.com/Mondo-Robotics/ECHO-G.git
cd ECHO-G
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional dependencies:

| Task | Install |
|---|---|
| Extract features for new raw inputs | `python -m pip install -e ".[preprocess]"` |
| Benchmark | `python -m pip install -e ".[benchmark]"` |
| MuJoCo rendering | `python -m pip install -e ".[visualization]"` |
| Development | `python -m pip install -e ".[dev]"` |

## Data and weights

**Release v0.2.0:** the dataset and all four model variants are publicly downloadable.
Use GitHub tag `v0.2.0` with HF revision `v0.2.0` for the matching code, data and weights.

| Asset | Repository / fixed revision | Contents |
|---|---|---|
| Dataset | [gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G/tree/v0.2.0) / `v0.2.0` | 14,987 training + 3,242 validation clips; motion, frozen conditions, audio, text, word times, splits, stats, BA normalization and G1 rendering assets |
| All model variants | [gaopusen/ECHO-G](https://huggingface.co/gaopusen/ECHO-G/tree/v0.2.0) / `v0.2.0` | Audio+text, audio-only, text-only, HumanRetarget and the shared FGD encoder |

Follow [the dataset guide](docs/DATASET.md#download-and-extract) to download and extract the data
into `data/echo-g`. Supplied frozen conditions are ready to use; no feature extraction is needed.

### Download model weights

For the audio+text examples below:

```bash
python -m pip install huggingface_hub
hf download gaopusen/ECHO-G --repo-type model \
  --revision v0.2.0 --local-dir weights \
  --include 'best.pt' 'evaluation/*' 'configs/sgdit_audio_text.yaml' \
            'SHA256SUMS' 'LICENSE*' 'NOTICE' 'README.md'
(cd weights && sha256sum --ignore-missing -c SHA256SUMS)
```

This downloads the audio+text files; the checksum command verifies files present locally.
For audio-only, text-only or HumanRetarget, use [the variant guide](docs/MODEL_VARIANTS.md).

## Inference

Generate the full validation split with the released audio+text checkpoint:

```bash
echo-g-sample \
  --checkpoint weights/best.pt \
  --config configs/sgdit_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir results/audio_text \
  --split val --seeds 0 --device cuda
```

For a quick check, add `--limit 3`; remove it for the full benchmark. Predictions are saved as
`results/audio_text/seed_000/<stem>.pt`. Each contains physical `robot_repr[T,39]` at 30 FPS.

For **your own audio/text or cached conditions**, follow [INFERENCE.md](docs/INFERENCE.md).
The guide includes frozen encoder downloads and clip-relative word timing. New inputs exceeding
the frame or token limit must be split explicitly.

## Training

Train the direct robot model using the supplied splits, frozen conditions and training statistics:

```bash
echo-g-train \
  --config configs/sgdit_audio_text.yaml \
  --data-root data/echo-g \
  --output-dir outputs/audio_text \
  --device cuda
```

Resume with `--resume outputs/audio_text/latest.pt`. `best.pt` is selected by EMA validation loss.
See [the training recipe and reference results](docs/REPRODUCIBILITY.md) for batch size, optimizer,
checkpoint selection and numerical reproducibility. Audio-only and text-only recipes are in
[MODEL_VARIANTS.md](docs/MODEL_VARIANTS.md#training-the-direct-robot-models).

## Benchmark and visualization

| Task | Entry point | Guide |
|---|---|---|
| FGD, Div, BA, weighted jerk, foot metrics and optional MM20 | `python scripts/eval_g1_motion_cls.py` | [BENCHMARK.md](docs/BENCHMARK.md) |
| Export predictions to MuJoCo NPZ | `echo-g-export-mujoco` | [VISUALIZATION.md](docs/VISUALIZATION.md) |
| Render NPZ to video and add audio | `echo-g-render`, then FFmpeg | [VISUALIZATION.md](docs/VISUALIZATION.md) |

The packaged audio+text implementation achieves **FGD 2.278311** on all 3,242 validation clips.
Historical checkpoint results are **FGD 2.278349 / MM20 1.785556**; MM20 was not rerun for the
package. [Reference results](docs/REPRODUCIBILITY.md#reference-results) specify the protocol.

## Repository layout

```text
configs/                      one configuration per model variant
src/echo_g/
  model.py                    speech-conditioned motion transformer
  word_time_attention.py      global/local text attention and signed time prior
  config.py, data.py          configuration and dataset loading
  conditions.py               raw audio/text feature extraction
  condition_provenance.py     complete-token and timestamp validation
  training.py, flow.py         training, rectified flow and sampling
  inference.py                paired-split and condition-only inference
  human_retarget/             HumanRetarget inference networks and decoding
  evaluation/                 G1 metrics, kinematics and learned FGD encoder
  visualization/              MuJoCo export and rendering
scripts/                      three thin evaluation/rendering entry points
docs/                         task guides, architecture and data/model cards
manifests/                    pinned asset versions, checksums and reference scores
tests/                        regression and integration tests
.github/                      CI and contributor/community guidance
```

### Documentation

| Need | Read |
|---|---|
| Download and understand the data | [Download](docs/DATASET.md) · [Data card](docs/DATASET_CARD.md) · [Tensor and timing format](docs/DATA_FORMAT.md) |
| Run new inputs or another model variant | [Inference](docs/INFERENCE.md) · [Model variants](docs/MODEL_VARIANTS.md) |
| Understand the model and reproduce results | [Architecture](docs/ARCHITECTURE.md) · [Model card](docs/MODEL_CARD.md) · [Reproduction](docs/REPRODUCIBILITY.md) |
| Evaluate or render predictions | [Benchmark](docs/BENCHMARK.md) · [MuJoCo visualization](docs/VISUALIZATION.md) |

## License and citation

Code uses [PolyForm Noncommercial 1.0.0](LICENSE). Project data contributions and weights use
[CC BY-NC 4.0](LICENSE-DATA-WEIGHTS); third-party materials retain their applicable terms.
See [licensing scope and contacts](docs/LICENSING.md), [NOTICE](NOTICE), and [CITATION.cff](CITATION.cff).

For development and bug reports, see [Contributing](.github/CONTRIBUTING.md).
