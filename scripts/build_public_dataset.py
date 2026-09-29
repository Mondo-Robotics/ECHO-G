# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from echo_g.condition_provenance import (
    FrozenConditionManifest,
    tensor_sha256,
    token_time_spans,
)

LOGGER = logging.getLogger(__name__)
SOURCE_VERSION = "610339f317ad3aa72a007cda14062cbdc39ca655930df14de1a1cb4b4007f4ad"
SOURCE_MANIFEST = "c4cc2b25a00483cf93fc06741cd1424ce5c14b53af9c72a02a9a7d91c72c7417"
SOURCE_PLAN = "17baab99180499b1f333cf0c85988e8cd1daccf465b7d8bddf22d5235d60dd0e"
BEAT_REVISION = "8689ecb43513ba31964fd60e0ca69be02d3b0872"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MOTION_CHECKSUMS_HASH = "41a5033d10e92acf9c8564e6c379961b09118f672ffc5adf36df3d92d1949b78"
MMAE_HASH = "0f763e6d54ca7bb33e49b0336ba2cc3359f9e9d5942e7f475c85c74b4b66e880"
SPLIT_HASHES = {
    "train_drop.txt": "18c9d6abdc1eabc79889d2d2cfc6620e75cb2781db9e0824473411b8dbdfe66a",
    "val_common.txt": "36a960134a321f7b2c47c1b24aabfdf1fa8ee213ea59a8570740be196ba7d83b",
}
MOTION_FIELDS = {
    "robot_repr",
    "real_num_frames",
    "fps",
    "source",
    "source_fps",
    "source_coordinate_system",
    "initial_yaw",
    "representation_schema",
    "coordinate_system",
    "root_forward_axis",
    "velocity_frame",
    "velocity_convention",
    "canonicalization",
}
CONDITION_FIELDS = {
    "audio_4fps",
    "text_pooled",
    "text_tokens",
    "token_times",
    "has_word_timing",
    "text_dim",
    "n_tokens",
    "dur",
    "stem",
    "latent_fps",
    "n_audio_50",
    "audio_dim",
}
STATS_FIELDS = {
    "schema_version",
    "mean",
    "std",
    "n_frames",
    "n_clips",
    "fps",
    "stats_stems_sha256",
    "normalization_scope",
    "source_values",
    "representation_schema",
    "coordinate_system",
    "root_forward_axis",
    "velocity_frame",
    "velocity_convention",
    "canonicalization",
}


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def public_metadata(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for v in value.values():
            public_metadata(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            public_metadata(v)
    elif isinstance(value, str):
        require(
            not value.startswith(("/data/", "/home/", "/root/", "/tmp/", "/opt/")),
            "Internal absolute path in public metadata",
        )


def write_bytes(path: Path, blob: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = digest(blob)
    if path.exists():
        require(not path.is_symlink(), f"Refusing symlink: {path}")
        require(file_hash(path) == expected, f"Existing output differs: {path}")
        return expected
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(blob)
    os.replace(temporary, path)
    require(file_hash(path) == expected, f"Write verification failed: {path}")
    return expected


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, value: Any) -> str:
    public_metadata(value)
    return write_bytes(path, json_bytes(value))


def repack(source: Path, target: Path, fields: set[str], removed: set[str]) -> dict[str, Any]:
    blob = source.read_bytes()
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), f"Invalid tensor payload: {source.name}")
    require(set(payload) == fields | removed, f"Unexpected fields in {source.name}: {set(payload)}")
    cleaned = {key: value for key, value in payload.items() if key in fields}
    public_metadata(cleaned)
    output = io.BytesIO()
    torch.save(cleaned, output)
    public_hash = write_bytes(target, output.getvalue())
    restored = torch.load(target, map_location="cpu", weights_only=True)
    require(set(restored) == fields, f"Public field mismatch: {target.name}")
    tensors = {}
    for key, value in cleaned.items():
        other = restored[key]
        if isinstance(value, torch.Tensor):
            require(
                value.dtype == other.dtype and value.shape == other.shape,
                f"Changed tensor layout: {target.name}:{key}",
            )
            require(torch.isfinite(value).all().item(), f"Nonfinite tensor: {source.name}:{key}")
            h = tensor_sha256(value)
            require(h == tensor_sha256(other), f"Changed tensor bytes: {target.name}:{key}")
            tensors[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": h}
        else:
            require(value == other, f"Changed numerical metadata: {target.name}:{key}")
    a, b = source.stat(), target.stat()
    require((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), "Public output aliases source inode")
    return {
        "source_sha256": digest(blob),
        "public_sha256": public_hash,
        "removed_fields": sorted(removed),
        "tensors": tensors,
        "all_retained_values_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an independently verified public V2 dataset copy"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--token-plan", type=Path, required=True)
    parser.add_argument("--source-motion-checksums", type=Path, required=True)
    parser.add_argument("--packaged", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mmae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limit", type=int, default=0, help="Canary only; does not produce a release manifest"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    root = args.output.resolve()
    source = args.source.resolve()
    require(root != source and source not in root.parents, "Use an independent output directory")
    require(args.workers >= 1 and args.limit >= 0, "Invalid workers/limit")
    require(
        file_hash(source / "VERSION.json") == SOURCE_VERSION, "Source VERSION identity mismatch"
    )
    require(file_hash(args.token_plan) == SOURCE_PLAN, "Independent token plan identity mismatch")
    require(
        file_hash(args.source_motion_checksums) == MOTION_CHECKSUMS_HASH,
        "Original motion checksum inventory identity mismatch",
    )
    motion_hashes = {}
    for line in args.source_motion_checksums.read_text().splitlines():
        expected_hash, relative = line.split(maxsplit=1)
        stem = Path(relative.strip()).stem
        require(stem not in motion_hashes, "Duplicate source motion identity")
        motion_hashes[stem] = expected_hash
    manifest = FrozenConditionManifest(args.source_manifest, SOURCE_MANIFEST)
    plan_rows = json.loads(args.token_plan.read_text())
    plans = {row["stem"]: row for row in plan_rows}
    require(len(plans) == len(plan_rows), "Duplicate token plan stems")
    split_stems: dict[str, list[str]] = {}
    for name, expected in SPLIT_HASHES.items():
        path = source / "splits" / name
        require(file_hash(path) == expected, f"Split identity mismatch: {name}")
        split_stems[name] = path.read_text().splitlines()
    train, val = split_stems["train_drop.txt"], split_stems["val_common.txt"]
    require(
        len(train) == 14987 and len(val) == 3242 and not set(train) & set(val), "Split mismatch"
    )
    stems = train + val
    require(set(stems) == set(plans) == set(manifest.entries), "Audit population mismatch")
    with (source / "audit/condition/per_clip_condition_motion_audit.csv").open() as stream:
        lengths = {row["stem"]: row for row in csv.DictReader(stream)}
    require(set(lengths) == set(stems), "Length audit population mismatch")
    for row in lengths.values():
        require(not row["error"], "Source length audit has errors")
    if args.limit:
        ordered = sorted(stems, key=lambda stem: plans[stem]["full_tokens"])
        candidates = [ordered[0], ordered[-1], *val[:2], *ordered[-3:]]
        stems = list(dict.fromkeys(candidates + stems))[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    train_set = set(train)
    root.mkdir(parents=True, exist_ok=True)

    def process(stem: str) -> dict[str, Any]:
        condition_path = source / "condition_30fps" / f"{stem}.pt"
        condition = torch.load(condition_path, map_location="cpu", weights_only=True)
        manifest.verify(condition_path, stem, condition)
        plan = plans[stem]
        package = args.packaged / stem
        metadata_path = package / "meta.json"
        require(
            file_hash(metadata_path) == plan["source_meta_sha256"], f"Changed source words: {stem}"
        )
        metadata = json.loads(metadata_path.read_text())
        words = [
            {"text": str(w["text"]), "start": float(w["start"]), "end": float(w["end"])}
            for w in metadata["words"]
        ]
        canonical = " ".join(word["text"] for word in words)
        transcript_blob = (package / "transcript.txt").read_bytes()
        require(
            transcript_blob.decode("utf-8").strip() == canonical, f"Transcript mismatch: {stem}"
        )
        encoding = tokenizer(canonical, truncation=False, return_offsets_mapping=True)
        ids = list(encoding["input_ids"])
        offsets = [list(pair) for pair in encoding["offset_mapping"]]
        ids_hash = tensor_sha256(torch.tensor(ids, dtype=torch.int64))
        require(ids_hash == plan["token_ids_sha256"], f"Tokenizer identity mismatch: {stem}")
        require(
            len(ids) == plan["full_tokens"] == condition["n_tokens"], f"Incomplete text: {stem}"
        )
        times = token_time_spans(canonical, offsets, words).to(torch.float16)
        require(
            tensor_sha256(times) == plan["full_token_times_sha256"],
            f"Word/token timing mismatch: {stem}",
        )
        require(torch.equal(times, condition["token_times"]), f"Frozen token times differ: {stem}")
        motion_target = root / "motion_39d_30fps" / f"{stem}.pt"
        condition_target = root / "condition_30fps" / f"{stem}.pt"
        motion_record = repack(
            source / "motion_39d_30fps" / f"{stem}.pt", motion_target, MOTION_FIELDS, {"source_npz"}
        )
        require(
            motion_record["source_sha256"] == motion_hashes.get(stem),
            f"Original motion identity mismatch: {stem}",
        )
        condition_record = repack(
            condition_path, condition_target, CONDITION_FIELDS, {"source_audio", "source_text_cond"}
        )
        annotation = {
            "schema": "echo-g-word-token-annotations-v1",
            "clip_id": stem,
            "time_origin": "start_of_packaged_audio",
            "time_unit": "seconds",
            "canonical_transcript": canonical,
            "words": words,
            "source": {
                "dataset": "H-Liu1997/BEAT2",
                "revision": BEAT_REVISION,
                "language": metadata["language"],
                "recording": metadata["source_recording"],
                "start_sec": metadata["start_sec"],
                "end_sec": metadata["end_sec"],
                "meta_sha256": plan["source_meta_sha256"],
            },
            "tokenization": {
                "model": "Qwen/Qwen3.5-4B",
                "revision": TOKENIZER_REVISION,
                "truncation": False,
                "full_token_count": len(ids),
                "token_ids": ids,
                "token_offsets": offsets,
                "token_ids_sha256": ids_hash,
                "token_times_sha256": tensor_sha256(times),
                "storage_dtype": "float16",
                "timing_rule": "character_offset_union_with_spaces_at_previous_word_end",
            },
        }
        annotation_hash = write_json(root / "annotations/words" / f"{stem}.json", annotation)
        transcript_hash = write_bytes(root / "transcripts" / f"{stem}.txt", transcript_blob)
        audio_hash = write_bytes(
            root / "audio" / f"{stem}.wav", (package / "audio.wav").read_bytes()
        )
        for source_name, target_path in [
            ("audio.wav", root / "audio" / f"{stem}.wav"),
            ("transcript.txt", root / "transcripts" / f"{stem}.txt"),
        ]:
            a, b = (package / source_name).stat(), target_path.stat()
            require(
                (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino),
                f"Media output aliases source inode: {stem}:{source_name}",
            )
        record = {
            "stem": stem,
            "split": "train" if stem in train_set else "val",
            "language": metadata["language"],
            "source_recording": metadata["source_recording"],
            "source_start_sec": metadata["start_sec"],
            "source_end_sec": metadata["end_sec"],
            "aligned_frames": int(lengths[stem]["aligned_frames"]),
            "full_token_count": len(ids),
            "paths": {
                "motion": f"motion_39d_30fps/{stem}.pt",
                "condition": f"condition_30fps/{stem}.pt",
                "audio": f"audio/{stem}.wav",
                "transcript": f"transcripts/{stem}.txt",
                "word_annotations": f"annotations/words/{stem}.json",
            },
            "sha256": {
                "motion": motion_record["public_sha256"],
                "condition": condition_record["public_sha256"],
                "audio": audio_hash,
                "transcript": transcript_hash,
                "word_annotations": annotation_hash,
            },
            "source_sha256": {
                "motion": motion_record["source_sha256"],
                "condition": condition_record["source_sha256"],
                "meta": plan["source_meta_sha256"],
            },
            "source_dataset": "H-Liu1997/BEAT2",
            "source_revision": BEAT_REVISION,
        }
        return {
            "clip": record,
            "motion": motion_record,
            "condition": condition_record,
            "manifest_entry": {
                **manifest.entries[stem],
                "condition_sha256": condition_record["public_sha256"],
            },
            "raw_input": {
                "stem": stem,
                "audio_path": f"audio/{stem}.wav",
                "transcript": canonical,
                "words": words,
            },
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(process, stems), start=1):
            records.append(row)
            if index == 1 or index % 500 == 0 or index == len(stems):
                LOGGER.info("Verified and copied %d/%d clips", index, len(stems))
    stats_path = source / "stats/drop_train.pt"
    require(
        file_hash(stats_path) == "9378214e1033a7fb309532a040b07f87a9010fbd484a30b8b9962cc75704aac8",
        "Source training statistics identity mismatch",
    )
    stats_record = repack(stats_path, root / "stats/drop_train.pt", STATS_FIELDS, {"gt_dir"})
    if args.limit:
        write_json(
            root / "CANARY.json", {"status": "pass", "clips": len(records), "stats": stats_record}
        )
        LOGGER.info("Canary completed; no release manifest written")
        return
    entries = {row["clip"]["stem"]: row["manifest_entry"] for row in records}
    result = {
        "schema": "echo-g-frozen-condition-manifest-v1",
        "entry_count": len(entries),
        "entries": entries,
        "source_release_version_sha256": SOURCE_VERSION,
        "source_condition_manifest_sha256": SOURCE_MANIFEST,
        "token_time_audit_sha256": SOURCE_PLAN,
        "token_time_audit_policy": (
            "Original audited counts and FP16 times; public words "
            "retokenized without truncation; all retained tensors bitwise equal"
        ),
        "packaging_policy": (
            "remove internal path fields; preserve all remaining tensors and metadata exactly"
        ),
    }
    manifest_hash = write_json(root / "audit/condition/frozen_conditions.json", result)
    for name in SPLIT_HASHES:
        write_bytes(root / "splits" / name, (source / "splits" / name).read_bytes())
    length_path = Path("audit/condition/per_clip_condition_motion_audit.csv")
    write_bytes(root / length_path, (source / length_path).read_bytes())
    require(file_hash(args.mmae) == MMAE_HASH, "Frozen MMAE identity mismatch")
    write_bytes(root / "eval_assets/mmae/g1_mmae_30body_30fps.npy", args.mmae.read_bytes())
    write_json(
        root / "eval_assets/mmae/manifest.json",
        {
            "schema": "echo-g-frozen-mmae-v1",
            "sha256": MMAE_HASH,
            "dtype": "float32",
            "shape": [30],
            "source_gt_clips": 20790,
            "source_gt_frames": 5014146,
            "scope": "frozen full-all corpus evaluation statistic; not train normalization",
            "method": (
                "frame-weighted body_speed_central; fk_positions_ba "
                "with identity root rotation and zero root translation"
            ),
        },
    )
    for name, values in [
        ("manifests/clips.jsonl", [row["clip"] for row in records]),
        ("raw_inputs.jsonl", [row["raw_input"] for row in records]),
        (
            "audit/repack_equivalence.jsonl",
            [
                {
                    "stem": row["clip"]["stem"],
                    "motion": row["motion"],
                    "condition": row["condition"],
                }
                for row in records
            ],
        ),
    ]:
        data = "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
        )
        write_bytes(root / name, data.encode())
    summary = {
        "schema": "echo-g-v2-public-repack-validation-v1",
        "status": "pass",
        "clips": len(records),
        "train_clips": len(train),
        "validation_clips": len(val),
        "maximum_full_token_count": max(row["clip"]["full_token_count"] for row in records),
        "validation_aligned_frames": sum(
            row["clip"]["aligned_frames"] for row in records if row["clip"]["split"] == "val"
        ),
        "all_retained_values_bitwise_equal": True,
        "all_word_token_maps_match_independent_audit": True,
        "source_inodes_not_modified_or_reused": True,
        "source_release_version_sha256": SOURCE_VERSION,
        "source_condition_manifest_sha256": SOURCE_MANIFEST,
        "public_condition_manifest_sha256": manifest_hash,
        "stats": stats_record,
        "media_upstream_audit": "separate full source comparison required before final packaging",
        "license_status": "component evidence prepared; final publication terms pending",
    }
    write_json(root / "audit/repack_summary.json", summary)
    write_json(
        root / "VERSION.json",
        {
            "schema": "echo-g-v2-public-dataset-v1",
            "version": "2026-09-29-rc1",
            "counts": {"train": len(train), "validation": len(val), "total": len(records)},
            "source_release_version_sha256": SOURCE_VERSION,
            "condition_manifest_sha256": manifest_hash,
            "max_tokens": summary["maximum_full_token_count"],
            "token_capacity": 256,
            "fps": 30,
            "max_valid_frames": 600,
            "source_dataset": "H-Liu1997/BEAT2",
            "source_revision": BEAT_REVISION,
            "publication_status": "staged_pending_final_validation_and_publication_terms",
        },
    )
    LOGGER.info("Public repack complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
