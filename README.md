# ECHO-G V2

**Embodied Co-speech Humanoid mOtion Generation**

ECHO-G V2 generates full-body Unitree G1 motion from audio and a word-timed transcript.
It generates physical 39D robot references at 30 FPS using a rectified-flow transformer.
V2 combines normalized Q/K content attention with global/local text attention and a learned,
directional Gaussian time prior.

```text
Audio -> frozen Wav2Vec2 -> 30 FPS acoustic features ---+
                                                      v
Gaussian motion noise -> motion + audio + positions -> V2 DiT -> physical robot39
                                                      ^
Word-timed text -> frozen Qwen -> token features -------+
                       global / local Gaussian cross-attention
```

This is the **V2 audio+text release candidate** on `release/v2-20260928`, organized as the
existing `src/echo_g` Python package. It is being reviewed before any change to `main`.
The earlier sinusoidal model is not the model distributed by this branch. Its archival
branch/tag will be arranged separately. HumanRetarget and Seedance data are outside this release.

## Data and weights

**The processed dataset and V2 weights will be released on Hugging Face. Links are pending.**
No Hugging Face repository IDs, revisions, or downloadable archives are available from this
release candidate yet.

| Asset | Hugging Face location | Status |
|---|---|---|
| V2 processed robot-motion dataset | **To be added** | Prepared; upload and release terms pending |
| V2 audio+text `best.pt` (15k, EMA) | **To be added** | Upload pending |
| Benchmark FGD encoder | **To be added** | Weight upload pending; identity recorded |
| Frozen BA normalization | With dataset | Included in prepared dataset |

The prepared dataset contains processed robot motions, frozen audio/text conditions, clip-aligned
audio, source transcripts, word-time annotations, frozen splits, training statistics, and the
frozen BA normalization array. The V2 split is **14,987 training clips + 3,242 validation clips**.
Audio is stored as `audio/<stem>.wav`, source text as `transcripts/<stem>.txt`, and canonical text,
word times, and tokenizer mapping as `annotations/words/<stem>.json`. `raw_inputs.jsonl` provides
inputs for feature extraction. BEAT2-derived audio/text retain their applicable source license
and attribution, with clipping/text reconstruction documented. Encoder weights and robot assets
remain separate downloads with their own terms. Seedance is not included.

The public benchmark reports FGD, Div, BA, weighted jerk, foot metrics, and optional MM20. It does
not require or distribute semantic labels. See [the benchmark protocol](docs/BENCHMARK.md).

- [V2 model card](docs/MODEL_CARD.md)
- [V2 dataset card](docs/DATASET_CARD.md)
- [Download and use the dataset](docs/DATASET.md)
- [Data format and timing contract](docs/DATA_FORMAT.md)
- [Dataset asset manifest](manifests/v2_dataset.json)
- [Frozen-condition manifest identity](manifests/v2_frozen_conditions.json)
- [Model asset manifest](manifests/v2_audio_text.json)
- [Benchmark asset manifest](manifests/benchmark_assets.json)

Follow the [dataset download guide](docs/DATASET.md), then use the commands below.
Hugging Face links will be filled in after publication.

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

## V2 at a glance

| Setting | V2 audio+text |
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

The original run completed 63,000 optimizer steps; **best15k and final63k are different
checkpoints**. V2 uses `frame_seconds - token_center_seconds`, preserving the sign needed for
learned lag. Text keys remain tokenizer units, including subwords; V2 does not pool them into
one key per word. See [the architecture](docs/V2_ARCHITECTURE.md).

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

The supplied V2 configurations use these files automatically. Download and extract the complete
dataset, then run:

```bash
echo-g-train \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root /path/to/v2_data \
  --output-dir outputs/v2_audio_text \
  --device cuda
```

Resume using the same command plus `--resume outputs/v2_audio_text/latest.pt`.
The training directory records `experiment.json`, `latest.pt`, `best.pt`, `final.pt`, and
`complete.json`. `latest.pt` is the full training-resume state; `best.pt` is selected by validation
loss, not by FGD. The training recipe is in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Inference

### Frozen validation split

Use the paired dataset entry point when reproducing the canonical benchmark:

```bash
echo-g-sample \
  --checkpoint /path/to/v2_best015000.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --data-root /path/to/v2_data \
  --output-dir outputs/v2_val \
  --split val --seeds 0 --device cuda
```

### Cached conditions without ground-truth motion

For new conditions produced by the current raw extractor, use the embedded complete-text
provenance:

```bash
echo-g-infer \
  --checkpoint /path/to/v2_best015000.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --condition-dir /path/to/condition_30fps \
  --output-dir outputs/v2_new_clips \
  --device cuda
```

For independent inference on released dataset conditions, supply the included condition manifest:

```bash
echo-g-infer \
  --checkpoint /path/to/v2_best015000.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --condition-dir /path/to/v2_data/condition_30fps \
  --condition-manifest /path/to/v2_data/audit/condition/frozen_conditions.json \
  --condition-manifest-sha256 81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01 \
  --output-dir outputs/v2_frozen_conditions \
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

Word times are relative to the audio clip. The V2 extractor reconstructs the canonical text
from the word list and preserves the original token/time mapping. Provide the frozen encoder
assets separately, then run:

```bash
echo-g-extract-conditions \
  --mode audio --manifest utterances.jsonl \
  --output-dir data/new_conditions \
  --audio-model /path/to/wav2vec2-large-xlsr-53-english \
  --fps 30 --device cuda

echo-g-extract-conditions \
  --mode text --manifest utterances.jsonl \
  --output-dir data/new_conditions \
  --text-model /path/to/Qwen3.5-4B \
  --text-layer -2 --max-text-tokens 256 --device cuda

echo-g-extract-conditions \
  --mode merge --manifest utterances.jsonl \
  --output-dir data/new_conditions

echo-g-infer \
  --checkpoint /path/to/v2_best015000.pt \
  --config configs/sgdit_v2_audio_text.yaml \
  --condition-dir data/new_conditions \
  --output-dir outputs/v2_raw \
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

Historical V2 best15k scores on common3242 were **FGD 2.278349** (seed000) and
**MM20 1.785556** (seeds 0–19). The packaged implementation scores **FGD 2.278311** under the released benchmark protocol;
MM20 refers to the original 20-seed run.
[Reproduction scope and reference results](docs/REPRODUCIBILITY.md#historical-results)
explain the distinction.

## Project layout

```text
configs/                  V2 training and inference configuration
src/echo_g/               model, data, training, inference, and condition extraction
scripts/                  G1 evaluation and MuJoCo visualization entry points
docs/                     architecture, data/model cards, and usage protocols
manifests/                asset identities and pending Hugging Face locations
tests/                    implementation and integration checks
```

## License and citation

This branch retains the existing [LICENSE](LICENSE) and [NOTICE](NOTICE) without changes.
Final code, dataset, weight, and third-party distribution terms will be discussed before
publication; a planned Hugging Face location is not a license grant. The current source license
is PolyForm Noncommercial 1.0.0. Citation metadata is in [CITATION.cff](CITATION.cff).
