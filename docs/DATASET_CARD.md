# ECHO-G V2 dataset card

Status: **release candidate, 2026-09-28; Hugging Face upload pending**.
This card describes the data used by V2 audio+text. Its source counts and identities are frozen;
public archive names, sizes, revisions, per-shard hashes, and final distribution terms remain to
be completed when the dataset is packaged. [Machine-readable manifest](../manifests/v2_dataset.json).

## Scope and population

The release targets co-speech motion generation in the Unitree G1 robot representation. It uses
processed BEAT2-derived motion and aligned frozen speech/text conditions. The first release
excludes Seedance, HumanRetarget targets, and the earlier all-status model/data configuration.

| Split | File | Clips |
|---|---|---:|
| Training, drop-joint-fail policy | `splits/train_drop.txt` | 14,987 |
| Speaker-held-out validation / common benchmark | `splits/val_common.txt` | 3,242 |
| Union of released model's train and validation splits | — | 18,229 |

The source data tree can contain additional all-status clips. Those clips are not part of this
V2 split. Never discover training examples by enumerating all condition files.

Validation holds out speaker prefixes `english_1_wayne_*`, `english_21_ayana_*`, and
`english_3_solomon_*`. The benchmark covers 3,242 clips and **891,351 aligned frames**. The
historical trainer's batch-three validation loader dropped its final incomplete batch and used
3,240 examples for checkpoint selection; full benchmark export covers all 3,242.

All model inputs use 30 FPS and at most 600 valid frames (20 seconds). V2 permits 256 tokenizer
units; the audited full-text release has at most 247, and common3242 at most 73. Token capacity is
not a word-count limit. The split, normalization, and text cache differ from older 64-token or
multi-rate releases.

## What will be distributed

| Component | Planned content |
|---|---|
| Processed robot motion | Per-clip physical 39D G1 motion and valid lengths |
| Frozen conditions | Aligned 1,024D acoustic features; complete 2,560D token features; token intervals |
| Word-time annotations | Clip-relative word intervals and reconstruction/provenance identifiers; text-bearing fields follow source terms |
| Splits and statistics | Exact train/validation stem lists and `drop_train.pt` |
| Audits and checksums | Alignment/quality provenance and packaged file/shard identities |

Raw speech, transcript text, source human-body assets, encoder weights, and robot assets are
separate dependencies. Their download or reconstruction instructions will be supplied according
to their distribution terms; they are not implicitly relicensed by the processed package.
Cached-condition training and motion inference do not require loading the raw waveform. BA
benchmark evaluation does require the corresponding waveform, and SRGR requires semantic labels.
See [benchmark requirements](BENCHMARK.md).

## Processing lineage

The source production chain starts with BEAT2 SMPL-X/AMASS-format motion at native 30 FPS:

1. GMR retargeting to 29-DoF G1 with the `posture_mild` setup.
2. Y-up to Z-up conversion; root stabilization where root translation J30 exceeds 80.
   The stabilization is Savitzky–Golay, window 9, polynomial order 3, applied to root XYZ and
   sign-continuous root quaternions; the 29 hinge-joint angles remain unchanged.
3. Clip-constant vertical grounding, then FK and derived velocity reconstruction at 30 FPS.
4. Physical-quality checks, explicit exclusions, and severe high-frequency-tail screening.
5. Direct construction of physical39 with frame-zero base yaw removed. This is a native 30 FPS
   chain, not the earlier 30→50→30 data pipeline.
6. Audio/motion length screening, condition completion, and frozen common-prefix alignment.
7. Train-only statistics, frozen split lists, and full-text condition preparation for V2.

Mandatory quality checks are Integrity, Foot Contact, Grounding, Self Collision, G1 Continuity,
Smoothness, and Match, all at Pass. The V2 `drop` policy additionally rejects Joint Limits Fail,
Unknown, and missing statuses, while accepting Pass/Warning. The severe-tail gate excludes clips
satisfying both world-space J30 ≥ 120 m/s³ and energy above 10 Hz ≥ 10%. The source release also
uses its frozen explicit-exclusion list; packaging must preserve the audited selection rather
than recomputing it with different scanner versions.

Final audio screening accepts only `audio_frames - motion_frames` in `[-1, +1]`. There is no
motion time-warp to audio length. Audited common-prefix lengths define paired loading. Before
that screen, the drop-policy pool had 18,315 clips; after it, the V2 train/validation union has
18,229. Frozen full-text preparation changes the text cache, not the split or motion statistics.

## Representation and normalization

The output representation is `projecthermes-g1-39d-standard-v2`: base orientation 6D, yaw delta,
yaw-local XYZ velocity, and 29 joint angles. Coordinates are Z-up, forward +X. See
[DATA_FORMAT.md](DATA_FORMAT.md) for tensors, units, timing, masks, and length limits.

The 39D normalization file was computed from 14,987 training clips / 3,537,311 physical frames,
using unbiased variance. It excludes validation. Audio and text encoders remain frozen.

## Frozen identity versus public archive identity

| Source artifact | SHA256 |
|---|---|
| Source full-text release VERSION artifact | `610339f317ad3aa72a007cda14062cbdc39ca655930df14de1a1cb4b4007f4ad` |
| `splits/train_drop.txt` | `18c9d6abdc1eabc79889d2d2cfc6620e75cb2781db9e0824473411b8dbdfe66a` |
| `splits/val_common.txt` | `36a960134a321f7b2c47c1b24aabfdf1fa8ee213ea59a8570740be196ba7d83b` |
| `stats/drop_train.pt` | `9378214e1033a7fb309532a040b07f87a9010fbd484a30b8b9962cc75704aac8` |
| Full-text readiness audit (`final_condition_audit.csv`) | `be4feba6a494d7175d5fc1243d17f4aa41d55d88fe9aa1a48866088b88130eb2` |
| Per-clip condition/motion length CSV | `af76c470398e4d63404e095d7d93637c4860e07a91229bc476227afcdeb7bbb2` |

These identify the model's original inputs. The source VERSION hash is not a hash of a future
Hugging Face archive or of this Markdown card. Repackaging must retain these identities and add
separate archive inventories/checksums. No archive checksum is claimed before packaging.

## Limitations and publication checklist

The data represent a curated source corpus, a particular retargeting and quality-selection
pipeline, a particular robot, and held-out speakers from that corpus. They do not establish
coverage of every speaker, language, gesture style, robot, or physical execution setting.
Generated references and MuJoCo renders are not a robot-control validation.

Before publishing data, fill in the Hugging Face repository/revision, shard inventories and
hashes, word-time annotation package schema, source retrieval/reconstruction instructions, and
component-specific terms. The code branch retains its existing license; the final publication
license discussion is deferred. These pending packaging items do not change the frozen V2
sample lists or reference experiment counts.
