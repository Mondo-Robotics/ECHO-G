# ECHO-G audio+text model card

Status: **uploaded private preview, 2026-09-30**, at
[gaopusen/ECHO-G](https://huggingface.co/gaopusen/ECHO-G/tree/2026-09-30). Access requires
an authorized account; public release remains pending.
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
run completed at 63,000 steps. The published file is an inference package with unrelated
training metadata and internal paths removed. Its identity differs from the original serialized
checkpoint; all inference parameters and normalization values are preserved.

| File at model revision `2026-09-30` | Bytes | SHA256 |
|---|---:|---|
| `best.pt` | 655,730,204 | `c84f31fce140cdc8c3be5e5dbc8cb8eb3a82ce1865e2d84026a0791d4735928d` |
| `evaluation/g1_fgd_encoder.pt` | 5,551,281 | `761d3ae1e833123765785c8d8fcb95ec3745ee07de230cc892b52c8ecd304b91` |

Use these release-file checksums to verify downloads. The weights contain the inference
parameters and normalization statistics; optimizer-resume state is not included.

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

Download the private model repository at revision `2026-09-30` using the
[README commands](../README.md#download-model-weights). It includes the main checkpoint, FGD
encoder, matching configuration, license and `SHA256SUMS`. Use dataset revision
`2026-09-29-license` from the [separate dataset repository](https://huggingface.co/datasets/gaopusen/ECHO-G/tree/2026-09-29-license).
Source speech, transcript and frozen audio/text encoder assets have separate terms. G1 XML and
meshes are included with the dataset under BSD-3-Clause; they are not embedded in this code repository.
The ECHO-G checkpoint is designated **CC BY-NC 4.0**, covering rights held by the project;
see [LICENSE-DATA-WEIGHTS](../LICENSE-DATA-WEIGHTS) and [licensing scope](LICENSING.md).
Code remains under PolyForm Noncommercial 1.0.0.
Third-party rights are not replaced by the checkpoint license.
