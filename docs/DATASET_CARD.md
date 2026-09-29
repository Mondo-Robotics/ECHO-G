# ECHO-G dataset card

Status: **uploaded private preview, 2026-09-29** at
[gaopusen/ECHO-G](https://huggingface.co/datasets/gaopusen/ECHO-G). Access is limited to
authorized accounts; public release remains pending.
This card describes the BEAT2-derived data used by ECHO-G.
[Download and usage](DATASET.md) · [Data format](DATA_FORMAT.md).

## Scope and population

The release targets co-speech motion generation in the Unitree G1 robot representation. It uses
processed BEAT2-derived motion and aligned frozen speech/text conditions. The first release
includes robot motion and excludes Seedance and human-motion targets.

| Split | File | Clips |
|---|---|---:|
| Training, drop-joint-fail policy | `splits/train_drop.txt` | 14,987 |
| Speaker-held-out validation / common benchmark | `splits/val_common.txt` | 3,242 |
| Union of released model's train and validation splits | — | 18,229 |

Use the supplied split lists to select examples.

Validation holds out speaker prefixes `english_1_wayne_*`, `english_21_ayana_*`, and
`english_3_solomon_*`. The benchmark covers 3,242 clips and **891,351 aligned frames**. The
historical trainer's batch-three validation loader dropped its final incomplete batch and used
3,240 examples for checkpoint selection; full benchmark export covers all 3,242.

All model inputs use 30 FPS and at most 600 valid frames (20 seconds). The model permits
256 tokenizer units; the dataset has at most 247, and the validation split at most 73.
Token capacity is not a word-count limit.

## Included data

| Component | Content |
|---|---|
| Processed robot motion | Per-clip physical 39D G1 motion and valid lengths |
| Frozen conditions | Aligned 1,024D acoustic features; complete 2,560D token features; token intervals |
| Clip-aligned audio | `audio/<stem>.wav`, original packaged waveform bytes and sample rate; no additional motion-length crop or time stretch |
| Source transcripts | `transcripts/<stem>.txt`, original packaged transcript bytes |
| Word-time and tokenizer annotations | `annotations/words/<stem>.json`, canonical transcript, clip-relative `words[text,start,end]`, token IDs/offsets and provenance |
| Raw-input manifest | `raw_inputs.jsonl`, paired waveform, transcript and word times for feature extraction |
| BA normalization | Frozen `eval_assets/mmae/g1_mmae_30body_30fps.npy`; shared reference for every compared model |
| Splits and statistics | Exact train/validation stem lists and `drop_train.pt` |
| Condition metadata | Canonical frame lengths and complete-text condition manifest used by the data loader |

The clip-aligned audio, transcripts and word annotations are included in the dataset under their
applicable BEAT2 source terms, with attribution and modification notices. They are not implicitly
relicensed as project-owned content. Original transcript bytes and the canonical text used for
tokenization are identified separately; do not substitute one silently for the other.
Source human-body models, encoder weights, and robot assets remain separate dependencies with
asset-specific download instructions. Cached-condition training and inference do not load the
waveform. BA evaluation uses the included audio and frozen normalization array. Semantic labels
are outside this release and are not needed for its [benchmark](BENCHMARK.md).

## Processing lineage

The source production chain starts with BEAT2 SMPL-X motion at native 30 FPS, converted into
an AMASS-compatible input format; this is not a claim that the source is the AMASS dataset:

1. GMR retargeting to 29-DoF G1 with the `posture_mild` setup.
2. Y-up to Z-up conversion; root stabilization where root translation J30 exceeds 80.
   The stabilization is Savitzky–Golay, window 9, polynomial order 3, applied to root XYZ and
   sign-continuous root quaternions; the 29 hinge-joint angles remain unchanged.
3. Clip-constant vertical grounding, then FK and derived velocity reconstruction at 30 FPS.
4. Physical-quality checks, explicit exclusions, and severe high-frequency-tail screening.
5. Direct construction of physical39 with frame-zero base yaw removed at native 30 FPS.
6. Audio/motion length screening, condition completion, and frozen common-prefix alignment.
7. Train-only statistics, frozen split lists, and full-text condition preparation.

Mandatory quality checks are Integrity, Foot Contact, Grounding, Self Collision, G1 Continuity,
Smoothness, and Match, all at Pass. The `drop` policy additionally rejects Joint Limits Fail,
Unknown, and missing statuses, while accepting Pass/Warning. The severe-tail gate excludes clips
satisfying both world-space J30 ≥ 120 m/s³ and energy above 10 Hz ≥ 10%. The source release also
uses a fixed explicit-exclusion list, reflected in the released split files.

Final audio screening accepts only `audio_frames - motion_frames` in `[-1, +1]`. There is no
motion time-warp to audio length. Stored common-prefix lengths define paired loading. Before
that screen, the drop-policy pool had 18,315 clips; after it, the train/validation union has
18,229. Frozen full-text preparation changes the text cache, not the split or motion statistics.

## Representation and normalization

The output representation is `projecthermes-g1-39d-standard-v2`: base orientation 6D, yaw delta,
yaw-local XYZ velocity, and 29 joint angles. Coordinates are Z-up, forward +X. See
[DATA_FORMAT.md](DATA_FORMAT.md) for tensors, units, timing, masks, and length limits.

The 39D normalization file was computed from 14,987 training clips / 3,537,311 physical frames,
using unbiased variance. It excludes validation. Audio and text encoders remain frozen.

## Audio and language coverage

The release contains 15,850 English and 2,379 Chinese clips from 1,856 BEAT2 recordings.
All WAVs are 16 kHz mono: 14,580 use PCM16 and 3,649 use floating-point encoding.
Audio retains the segmented source waveforms, and word times are relative to each clip.

## Limitations and terms

The data represent a curated source corpus, a particular retargeting and quality-selection
pipeline, a particular robot, and held-out speakers from that corpus. They do not establish
coverage of every speaker, language, gesture style, robot, or physical execution setting.
Generated references and MuJoCo renders are not a robot-control validation.

BEAT2-derived audio/text retain their applicable upstream terms and attribution. The dataset
includes source and modification notices. The license for new contributions will be finalized
before public release; private preview access does not grant additional distribution rights.
The code branch retains its existing license. The pinned dataset revision is recorded in the
[dataset manifest](../manifests/v2_dataset.json).
