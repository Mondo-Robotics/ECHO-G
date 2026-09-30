# Inference on your own inputs

Start with the [installation and weight download](../README.md#installation).
For the supplied validation split, use the [README sampling command](../README.md#inference).
This guide covers cached conditions and new audio with word-timed text.
Audio-only, text-only, and HumanRetarget commands are in [MODEL_VARIANTS.md](MODEL_VARIANTS.md).

## Cached conditions without ground-truth motion


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

## Raw audio and word-timed text

The dataset includes `raw_inputs.jsonl` for its released clips. For your own inputs, create
`utterances.jsonl` with paths relative to that manifest:

```json
{"stem":"example_0001","audio_path":"audio/example.wav","transcript":"Hello world","words":[{"text":"Hello","start":0.10,"end":0.42},{"text":"world","start":0.55,"end":0.91}]}
```

Word times are relative to the audio clip. The ECHO-G extractor reconstructs the canonical text
from the word list and preserves the original token/time mapping.

### Download the frozen encoders

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
[the encoder manifest](../manifests/condition_encoders.json).

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

### Extract and infer

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
split, truncate, or stitch them. See [the timing and length rules](DATA_FORMAT.md#time-and-length).

Use frozen conditions for historical benchmark reproduction. Fresh encoder outputs can vary
with encoder revisions, decoding, hardware, and backend; exact agreement with the historical
audio or text cache is not claimed.
