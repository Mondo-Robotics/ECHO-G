# V2 public dataset packaging — 2026-09-29

The release candidate contains **18,229 clips**: 14,987 training and 3,242 validation.
Motion, frozen conditions, segmented WAV, transcript and word/token timing are paired per clip.
The original training source remains unchanged. Seedance, HumanRetarget, semantic/SRGR caches,
encoder weights and robot/body model assets are outside this dataset.

## Prepared artifacts

- 12 uncompressed USTAR archives; 91,145 per-clip members; 18,523,729,920 archive bytes.
- Root metadata: splits, training-only statistics, frozen MMAE, provenance and license notices,
  canonical length audit, clip mappings, acceptance reports and expanded-file SHA256 checksums.
- [Archive names, sizes and SHA256 values](../manifests/v2_dataset.json).
- Dataset-level license is not assigned while the owners decide the terms for new contributions.
  Upstream BEAT2 and encoder evidence is recorded separately in the dataset's third-party notices.

The HF destination and immutable revision remain pending. These are prepared local archives;
this document does not claim they have been uploaded or tested by downloading from HF.

## Download and unpack after publication

Replace the two placeholders with the published repository and immutable revision:

```bash
hf download <HF_DATASET_REPO> --repo-type dataset \
  --revision <PINNED_COMMIT_SHA> --local-dir downloads/echo-g-v2

python scripts/unpack_dataset.py \
  --download-dir downloads/echo-g-v2 --output data/echo-g-v2
```

Use an ordinary local directory from `--local-dir`; cached snapshot symlinks are rejected.
The HF directory contains `data/*.tar` and metadata. Unpack before passing it as DATA_ROOT.
The helper verifies all archive hashes and member inventories, rejects unsafe paths and links,
preserves different existing files, and verifies the expanded `checksums/SHA256SUMS` inventory.
The checksum file itself and archive transport indexes are identified separately.

```bash
echo-g-validate-data --config configs/sgdit_v2_audio_text.yaml \
  --data-root data/echo-g-v2
```

## Acceptance scope

[Public acceptance summary](../validation/public_dataset_acceptance.json):

| Check | Result |
|---|---|
| Source WAV, transcript and word times | All 18,229 clips exactly match the selected official source intervals |
| Source-to-public retained tensors and metadata | All 18,229 clips and training statistics exact |
| Independent full word-to-token mapping | All 18,229 clips passed; capacity 256, observed maximum 247 |
| Public strict loader | All 18,229 clips passed |
| Inference parity | Five distinct mixed-language/length clips, plus one singleton repeat; six exact matched-input comparisons |
| Benchmark on public GT/audio | 3,242 clips / 891,351 frames; every retained field equals the preceding acceptance |
| MuJoCo | Real G1, 312 frames, 960×720 at 30 FPS; video/audio both 10.4 seconds |
| Full archive roundtrip | 12 shards extracted into a separate empty directory; all 91,174 expanded-file checksums passed |

The benchmark **reuses immutable predictions from the preceding full inference acceptance**.
Only the five selected public-input clips were generated for this packaging acceptance.
FGD is 2.2783108333534123. SRGR is excluded and MM20 was not rerun.
This checks packaging equivalence; it does not establish fresh-encoder equivalence or new
full-set generation. [Earlier implementation acceptance](BRANCH_VALIDATION.md) retains its
historical scope and results separately.

Maintainer tools: `scripts/build_public_dataset.py` verifies pinned source identities and exports
an independent sanitized copy; `scripts/package_dataset_archives.py` creates and verifies the
12 data shards. Both require the audited source files; they do not download upstream data.

[Full archive roundtrip report](../validation/dataset_archive_roundtrip.json). This validates the
local upload candidate, not an HF download. Transport indexes and the checksum file itself
are copied with independent source/destination hash checks.
