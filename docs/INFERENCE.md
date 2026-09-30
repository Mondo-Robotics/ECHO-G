# Inference

Start with the [installation](../README.md#installation) and extract the
[dataset](https://huggingface.co/datasets/gaopusen/ECHO-G#download) into `data/echo-g`.
All commands below run from the ECHO-G checkout. The four models output physical
39-channel G1 motion at 30 FPS; HumanRetarget supports inference only.

## Model weights

The [README](../README.md#download-model-weights) downloads audio + text. For other variants:

```bash
hf download gaopusen/ECHO-G --revision v0.2.0 --local-dir weights \
  --include 'audio_only/*' 'text_only/*' 'human_retarget/*' \
            'configs/*' 'evaluation/*' 'SHA256SUMS' 'LICENSE*' 'NOTICE'
(cd weights && sha256sum --ignore-missing -c SHA256SUMS)
```

Keep only the variant directory patterns you need. HumanRetarget requires all three
files in its directory: `best.pt`, `retarget_vae.pt`, and `human_motion_vae.pt`.
File identities are recorded in [assets.json](../manifests/assets.json).

## Dataset clips

The [quick start](../README.md#quick-start) runs audio + text. Other variants use:

```bash
echo-g-sample --config configs/sgdit_audio_only.yaml \
  --checkpoint weights/audio_only/best.pt \
  --data-root data/echo-g --output-dir results/audio_only --limit 3 --device cuda

echo-g-sample --config configs/sgdit_text_only.yaml \
  --checkpoint weights/text_only/best.pt \
  --data-root data/echo-g --output-dir results/text_only --limit 3 --device cuda

echo-g-sample --config configs/human_retarget_inference.yaml \
  --checkpoint weights/human_retarget/best.pt \
  --retarget-checkpoint weights/human_retarget/retarget_vae.pt \
  --human-vae-checkpoint weights/human_retarget/human_motion_vae.pt \
  --data-root data/echo-g --output-dir results/human_retarget --limit 3 --device cuda
```

Remove `--limit 3` to generate the full validation split. Defaults are seed 0, EMA,
8 Euler steps, CFG 1 and batch size 3. Predictions in `seed_000/*.pt` contain
`robot_repr`, valid frame count, FPS and physical units. All variants share the
[evaluator](BENCHMARK.md) and [renderer](VISUALIZATION.md).

## Cached conditions without ground-truth motion

For released frozen conditions, supply both manifest options and the canonical length CSV:

```bash
echo-g-infer --config configs/sgdit_audio_text.yaml --checkpoint weights/best.pt \
  --condition-dir data/echo-g/condition_30fps \
  --condition-manifest data/echo-g/audit/condition/frozen_conditions.json \
  --condition-manifest-sha256 81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01 \
  --lengths-csv data/echo-g/audit/condition/per_clip_condition_motion_audit.csv \
  --stem-list data/echo-g/splits/val_common.txt \
  --output-dir results/cached --device cuda
```

`echo-g-infer` uses explicit manifest arguments, rather than resolving a dataset root from
YAML. For new conditions created by the extractor below, use their embedded provenance and
omit the two manifest options and length CSV. Without a length CSV, the acoustic sequence
sets the output length. The same config/checkpoint and HumanRetarget decoder options apply
as for `echo-g-sample`.

## Download the frozen encoders

Only new raw inputs need encoder downloads. Install preprocessing dependencies and download
the pinned upstream models (about 10.6 GB); their own licenses apply:

```bash
python -m pip install -e '.[preprocess]' huggingface_hub
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir models/Qwen3.5-4B
hf download jonatasgrosman/wav2vec2-large-xlsr-53-english \
  config.json preprocessor_config.json special_tokens_map.json vocab.json model.safetensors README.md \
  --revision 569a6236e92bd5f7652a0420bfe9bb94c5664080 \
  --local-dir models/wav2vec2-large-xlsr-53-english
sha256sum -c manifests/condition_encoders.sha256
```

Use separate local model directories and verify each listed file. Encoder classes, layers and
file hashes are in [condition_encoders.json](../manifests/condition_encoders.json).
Audio-only needs Wav2Vec2; text-only needs Qwen. If downloading just one encoder, check its
files against the corresponding entries in the manifest.

## New audio and word-timed text

Create `utterances.jsonl`, with one request per line and audio paths relative to that file:

```json
{"stem":"example","audio_path":"audio/example.wav","transcript":"Hello world","words":[{"text":"Hello","start":0.10,"end":0.42},{"text":"world","start":0.55,"end":0.91}]}
```

Word times are in seconds from the start of the clip. The extractor forms canonical text
from the word list and preserves the word-to-token mapping. The dataset's `raw_inputs.jsonl`
uses the same request format.

```bash
echo-g-extract-conditions --mode audio --manifest utterances.jsonl \
  --audio-model models/wav2vec2-large-xlsr-53-english \
  --output-dir data/new_conditions --fps 30 --device cuda
echo-g-extract-conditions --mode text --manifest utterances.jsonl \
  --text-model models/Qwen3.5-4B --text-layer -2 --max-text-tokens 256 \
  --output-dir data/new_conditions --device cuda
echo-g-extract-conditions --mode merge --manifest utterances.jsonl \
  --output-dir data/new_conditions
echo-g-infer --config configs/sgdit_audio_text.yaml --checkpoint weights/best.pt \
  --condition-dir data/new_conditions --output-dir results/new_audio_text --device cuda
```

For HumanRetarget, use its config and generator checkpoint plus the two decoder checkpoint
options shown above. No human training dataset or SMPL-X installation is needed.

### Audio only

Create `audio_requests.jsonl` without transcript fields:

```json
{"stem":"example","audio_path":"audio/example.wav"}
```

```bash
echo-g-extract-conditions --mode audio --manifest audio_requests.jsonl \
  --audio-model models/wav2vec2-large-xlsr-53-english --output-dir data/new_audio --device cuda
echo-g-infer --config configs/sgdit_audio_only.yaml --checkpoint weights/audio_only/best.pt \
  --condition-dir data/new_audio/.audio --output-dir results/new_audio --device cuda
```

### Text only

Create `text_requests.jsonl` with an explicit duration and clip-relative word times:

```json
{"stem":"example","duration":2.0,"words":[{"text":"Hello","start":0.2,"end":0.6},{"text":"there.","start":0.7,"end":1.2}]}
```

```bash
echo-g-extract-conditions --mode text --manifest text_requests.jsonl \
  --text-model models/Qwen3.5-4B --output-dir data/new_text --device cuda
echo-g-infer --config configs/sgdit_text_only.yaml --checkpoint weights/text_only/best.pt \
  --condition-dir data/new_text/.text --output-dir results/new_text --device cuda
```

The text path supplies zero acoustic features and does not run the audio encoder.
Word times must fit the requested duration.

## Length and reproducibility

New inputs use `T = max(2, round(duration_seconds * 30))`, from audio duration or the explicit
text-only duration. Inputs over 20 seconds or 256 tokens are rejected; split them explicitly
and make timestamps relative to each new clip. See [timing rules](DATA_FORMAT.md#time-and-length).
Use the released frozen conditions for reference benchmarks: fresh extraction can differ
numerically across encoders and runtime environments.
