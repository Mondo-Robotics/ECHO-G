# ECHO-G

**Embodied Co-speech Humanoid mOtion Generation**

Generate full-body Unitree G1 robot motion from speech and word-timed text.
Outputs are physical 39-channel motion at 30 FPS, with up to **600 frames / 20 seconds**
and **256 text tokens** per input.

| Model | Configuration | Support |
|---|---|---|
| Audio + text | [sgdit_audio_text.yaml](configs/sgdit_audio_text.yaml) | Training and inference |
| Audio only | [sgdit_audio_only.yaml](configs/sgdit_audio_only.yaml) | Training and inference |
| Text only | [sgdit_text_only.yaml](configs/sgdit_text_only.yaml) | Training and inference |
| HumanRetarget | [human_retarget_inference.yaml](configs/human_retarget_inference.yaml) | Inference only |

## Installation

Use Python 3.10 or 3.11 with a compatible PyTorch installation:

```bash
git clone https://github.com/Mondo-Robotics/ECHO-G.git
cd ECHO-G
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . huggingface_hub
```

## Data and weights

- [Dataset and extraction instructions](https://huggingface.co/datasets/gaopusen/ECHO-G#download):
  14,987 training and 3,242 validation clips, including ready-to-use frozen conditions.
  Extract into `data/echo-g`.
- [Model weights](https://huggingface.co/gaopusen/ECHO-G): all four variants and the shared FGD encoder.

Downloads use HF revision `v0.2.0`; the matching code snapshot is tagged
[v0.2.0](https://github.com/Mondo-Robotics/ECHO-G/tree/v0.2.0).

### Download model weights

For audio + text:

```bash
hf download gaopusen/ECHO-G --revision v0.2.0 --local-dir weights \
  --include 'best.pt' 'evaluation/*' 'configs/sgdit_audio_text.yaml' \
            'SHA256SUMS' 'LICENSE*' 'NOTICE' 'README.md'
(cd weights && sha256sum --ignore-missing -c SHA256SUMS)
```

## Quick start

Generate three validation clips:

```bash
echo-g-sample --checkpoint weights/best.pt \
  --config configs/sgdit_audio_text.yaml --data-root data/echo-g \
  --output-dir results/quick_start --split val --seeds 0 --limit 3 --device cuda
```

Predictions appear in `results/quick_start/seed_000/<stem>.pt` as physical
`robot_repr[T,39]`. For your own inputs or another model, follow the inference guide.

## Guides

| Task | Documentation |
|---|---|
| Run any model on dataset clips or new inputs | [Inference](docs/INFERENCE.md) |
| Train or resume the three direct robot models | [Training](docs/TRAINING.md) |
| Reproduce FGD, BA, jerk, foot metrics and MM20 | [Benchmark and reference scores](docs/BENCHMARK.md) |
| Render motion with MuJoCo and add audio | [Visualization](docs/VISUALIZATION.md) |
| Understand tensors, normalization and word/token times | [Data format](docs/DATA_FORMAT.md) |
| Understand attention and HumanRetarget decoding | [Architecture](docs/ARCHITECTURE.md) |

`src/echo_g/` contains the implementation; `configs/` contains one configuration per model.
`scripts/` provides evaluation/rendering entry points, `manifests/` pins assets and reference
scores, and `tests/` contains regression tests. See [Contributing](.github/CONTRIBUTING.md)
for development checks.

## License and citation

Code: [PolyForm Noncommercial 1.0.0](LICENSE).
Project data contributions and weights: [CC BY-NC 4.0](LICENSE-DATA-WEIGHTS).
Third-party materials retain their applicable terms. See [NOTICE](NOTICE) for scope,
attribution and contacts, and [CITATION.cff](CITATION.cff) for citation metadata.
