# ECHO-G audio+text model card

Status: **release candidate; weights upload to Hugging Face pending**.
[Model manifest](../manifests/audio_text.json) · [Data card](DATASET_CARD.md) ·
[Architecture](ARCHITECTURE.md) · [Reproduction protocol](REPRODUCIBILITY.md).

## Intended use and interface

ECHO-G generates co-speech G1 robot motion for research, evaluation, and offline visualization.
Inputs are a complete audio clip plus word-timed transcript, or frozen features representing
those inputs. Outputs are 30 FPS, unnormalized physical39 robot references. Each input supports
at most 20 seconds / 600 frames and 256 tokenizer units. Word timing must use clip-relative
seconds. It does not infer utterance duration from text alone or perform automatic long-form
segmentation/stitching.

The model uses 12 transformer blocks, learned positions, QKNorm, and global/local Gaussian token
attention. The local prior has learned per-head width, signed lag, and mixing coefficient. It is
a stochastic rectified-flow model, with repeatable fixed-seed Euler sampling. No human-motion VAE
or retarget decoder is required for this direct robot model.

## Weight identity and training data

The reference `best.pt` is EMA at step 15,000, selected by minimum validation loss from a training
run completed at 63,000 steps. SHA256:

```text
c1e7e863e28b76b710bc18f9e9029836771fc86c1bb9fca42aa9d7655f74a616
```

Training uses the 14,987-clip BEAT2-derived G1 full-text drop split; evaluation uses 3,242 held-out
clips. Data and normalization hashes are part of the release manifest.

## Evaluation and limitations

The packaged implementation scores **FGD 2.278311** on common3242 (3,242 clips,
891,351 frames). The reference checkpoint results are FGD 2.278349 (seed000) and
MM20 1.785556 (seeds 0–19). The full reference table and sampling settings are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#reference-results).

The model is conditioned on curated data and the specific frozen encoders. Different text
construction, timestamps, feature extraction stacks, or model assets can change outputs.
Tokenizer states are not word-pooled, and raw audio/text feature re-extraction is not guaranteed to
match the historical cache numerically. Evaluation on these held-out speakers does not establish
performance across arbitrary voices, languages, robots, or physical robot control settings.
MuJoCo visualization is a rendering of reference motion, not a closed-loop tracking validation.

## Distribution

The model's Hugging Face URL and immutable revision will be added after upload. Source speech,
transcript and encoder assets have separate terms. G1 XML and meshes are included with the
dataset under BSD-3-Clause; they are not embedded in this code repository.
Final release licensing remains to be discussed; the existing repository license is unchanged.
