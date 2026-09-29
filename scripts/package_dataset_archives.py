# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
COMPONENTS = {
    "motion": "motion_39d_30fps",
    "condition": "condition_30fps",
    "audio": "audio",
    "transcript": "transcripts",
    "word_annotations": "annotations",
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_archive(path: Path, expected: dict[str, str]) -> None:
    seen = set()
    with tarfile.open(path, "r|") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            h = hashlib.sha256()
            for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                h.update(block)
            if h.hexdigest() != expected[member.name]:
                raise ValueError(f"Archived content differs: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive population mismatch")


def write_archive(
    root: Path, output: Path, names: list[str], hashes: dict[str, str]
) -> dict[str, Any]:
    expected = {name: hashes[name] for name in names}
    if not output.exists():
        temporary = output.with_suffix(".tar.partial")
        if temporary.exists():
            raise FileExistsError(f"Remove or inspect incomplete archive first: {temporary}")
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for name in names:
                source = root / name
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"Only independent regular files can be archived: {name}")
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        verify_archive(temporary, expected)
        os.replace(temporary, output)
    else:
        verify_archive(output, expected)
    row = {
        "path": "data/" + output.name,
        "bytes": output.stat().st_size,
        "sha256": digest_file(output),
        "files": len(names),
        "expanded_bytes": sum((root / name).stat().st_size for name in names),
        "member_paths": names,
        "all_member_hashes_verified": True,
    }
    LOGGER.info("Verified %s: %d files, %.3f GB", output.name, len(names), row["bytes"] / 1e9)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, content-verified dataset shards"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.max_shard_bytes < 1024 * 1024:
        raise ValueError("Shard size must be at least 1 MiB")
    root = args.data_root.resolve()
    output = args.output.resolve()
    if root == output or root in output.parents:
        raise ValueError("Archive output must be separate from the expanded dataset")
    clips = [json.loads(line) for line in (root / "manifests/clips.jsonl").read_text().splitlines()]
    summary = json.loads((root / "audit/repack_summary.json").read_text())
    media = json.loads((root / "audit/media_sources_summary.json").read_text())
    if summary["status"] != "pass" or media["status"] != "pass":
        raise ValueError("Require successful full repack and source audit first")
    hashes = {}
    for clip in clips:
        for component in COMPONENTS:
            path = clip["paths"][component]
            if path in hashes:
                raise ValueError(f"Duplicate public path: {path}")
            hashes[path] = clip["sha256"][component]
    (output / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for component, directory in COMPONENTS.items():
        names = sorted(clip["paths"][component] for clip in clips)
        actual = {str(p.relative_to(root)) for p in (root / directory).rglob("*") if p.is_file()}
        if actual != set(names):
            raise ValueError(f"Unexpected or missing {directory} files")
        groups: list[list[str]] = []
        current: list[str] = []
        size = 10240
        for name in names:
            payload_size = (root / name).stat().st_size
            tar_size = 512 + ((payload_size + 511) // 512) * 512
            if tar_size + 10240 > args.max_shard_bytes:
                raise ValueError(f"File exceeds shard limit: {name}")
            if current and size + tar_size > args.max_shard_bytes:
                groups.append(current)
                current, size = [], 10240
            current.append(name)
            size += tar_size
        if current:
            groups.append(current)
        for index, members in enumerate(groups):
            target = output / "data" / f"{directory}-{index:05d}-of-{len(groups):05d}.tar"
            rows.append(write_archive(root, target, members, hashes))
    result = {
        "schema": "echo-g-dataset-shards-v1",
        "format": "tar/ustar",
        "compression": "none",
        "max_shard_bytes": args.max_shard_bytes,
        "clips": len(clips),
        "files": sum(row["files"] for row in rows),
        "archive_bytes": sum(row["bytes"] for row in rows),
        "expanded_bytes": sum(row["expanded_bytes"] for row in rows),
        "clip_manifest_sha256": digest_file(root / "manifests/clips.jsonl"),
        "shards": rows,
    }
    (output / "data_archives.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "ARCHIVE_SHA256SUMS").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows)
    )
    LOGGER.info("All archives verified: %d files in %d shards", result["files"], len(rows))


if __name__ == "__main__":
    main()
