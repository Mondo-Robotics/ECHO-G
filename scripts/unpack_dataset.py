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
import re
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

LOGGER = logging.getLogger(__name__)
GLOBAL_SUMS = "checksums/SHA256SUMS"
BLOCK_SIZE = 1024 * 1024


def relative_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"Unsafe or noncanonical relative path: {value!r}")
    return value


def no_links(path: Path) -> None:
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {part}")


def regular_file(path: Path) -> None:
    no_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular file: {path}")


def file_hash(path: Path) -> str:
    regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA256: {value!r}")
    return value


def read_checksums(path: Path) -> dict[str, str]:
    regular_file(path)
    result: dict[str, str] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError(f"{path}:{index}: invalid SHA256SUMS record")
        name = relative_path(match[2])
        if name in result:
            raise ValueError(f"{path}: duplicate checksum path {name}")
        result[name] = match[1]
    if not result:
        raise ValueError(f"Empty checksum inventory: {path}")
    return result


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def metadata_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        directories[:] = sorted(
            name
            for name in directories
            if name not in {".cache", ".git"} and not (parent == root and name == "data")
        )
        for name in directories:
            no_links(parent / name)
        for name in sorted(filenames):
            if name in {".cache", ".git"} or (parent == root and name == "data"):
                continue
            path = parent / name
            regular_file(path)
            result[relative_path(path.relative_to(root).as_posix())] = path
    return result


def validate_member(member: tarfile.TarInfo, allowed: set[str], seen: set[str]) -> str:
    name = relative_path(member.name)
    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.sparse is not None:
        raise ValueError(f"Archive member is not an ordinary file (links forbidden): {name}")
    if member.size < 0 or name not in allowed or name in seen:
        raise ValueError(f"Unexpected, duplicate, or invalid archive member: {name}")
    seen.add(name)
    return name


def archive_plan(root: Path, checksums: dict[str, str]) -> list[dict[str, Any]]:
    manifest_path = root / "data_archives.json"
    regular_file(manifest_path)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
    )
    if not isinstance(manifest, dict) or manifest.get("schema") != "echo-g-dataset-shards-v1":
        raise ValueError("Unsupported data_archives.json schema")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Archive manifest has no shards")
    paths: set[str] = set()
    members: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in shards:
        name = relative_path(record["path"])
        if not name.startswith("data/") or not name.endswith(".tar") or name in paths:
            raise ValueError(f"Invalid or repeated archive path: {name}")
        digest = sha256_value(record["sha256"])
        if checksums.get(name) != digest:
            raise ValueError(f"Archive checksum inventories disagree: {name}")
        names = record["member_paths"]
        if not isinstance(names, list) or not names:
            raise ValueError(f"Missing member inventory: {name}")
        allowed = {relative_path(item) for item in names}
        if len(allowed) != len(names) or members.intersection(allowed):
            raise ValueError(f"Repeated member paths in archive manifest: {name}")
        if type(record["files"]) is not int or record["files"] != len(names):
            raise ValueError(f"Wrong file count in archive manifest: {name}")
        archive = root / name
        regular_file(archive)
        if type(record["bytes"]) is not int or archive.stat().st_size != record["bytes"]:
            raise ValueError(f"Archive size mismatch: {name}")
        if file_hash(archive) != digest:
            raise ValueError(f"Archive SHA256 mismatch: {name}")
        seen: set[str] = set()
        expanded_bytes = 0
        with tarfile.open(archive, mode="r:", ignore_zeros=True) as stream:
            for member in stream:
                validate_member(member, allowed, seen)
                expanded_bytes += member.size
        if seen != allowed:
            raise ValueError(f"Archive is missing listed members: {name}")
        if "expanded_bytes" in record and expanded_bytes != record["expanded_bytes"]:
            raise ValueError(f"Archive expanded size mismatch: {name}")
        paths.add(name)
        members.update(allowed)
        result.append({"path": archive, "members": allowed, "stat": archive.stat()})
        LOGGER.info("Verified archive %d/%d: %s", len(result), len(shards), name)
    if set(checksums) != paths:
        raise ValueError("ARCHIVE_SHA256SUMS contains archives outside data_archives.json")
    if manifest.get("files", len(members)) != len(members):
        raise ValueError("Global archive file count mismatch")
    return result


def check_destinations(output: Path, names: set[str], expected: dict[str, str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        if any(str(parent) in names for parent in path.parents if str(parent) != "."):
            raise ValueError(f"File/directory path collision: {name}")
        target = output / name
        no_links(target)
        if target.exists() and file_hash(target) != expected[name]:
            raise ValueError(f"Refusing to overwrite different existing content: {target}")


def copy_verified(
    source: BinaryIO,
    output: Path,
    name: str,
    size: int,
    expected: str,
) -> bool:
    target = output / name
    no_links(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.exists()
    temporary: Path | None = None
    sink = None
    try:
        if not existing:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".echo-g-unpack-", dir=target.parent
            )
            temporary = Path(temporary_name)
            sink = os.fdopen(descriptor, "wb")
        digest = hashlib.sha256()
        copied = 0
        for chunk in iter(lambda: source.read(BLOCK_SIZE), b""):
            copied += len(chunk)
            if copied > size:
                raise ValueError(f"Source grew during extraction: {name}")
            digest.update(chunk)
            if sink is not None:
                sink.write(chunk)
        if copied != size or digest.hexdigest() != expected:
            raise ValueError(f"Expanded content checksum/size mismatch: {name}")
        if sink is not None:
            sink.close()
            sink = None
            assert temporary is not None
            temporary.chmod(0o644)
            try:
                os.link(temporary, target)
            except FileExistsError:
                if file_hash(target) != expected:
                    raise ValueError(f"Refusing to overwrite changed content: {target}") from None
                return False
            return True
        if file_hash(target) != expected:
            raise ValueError(f"Existing content changed during extraction: {target}")
        return False
    finally:
        if sink is not None:
            sink.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack(download_dir: Path, output: Path) -> dict[str, Any]:
    root = download_dir.absolute()
    output = output.absolute()
    no_links(root)
    no_links(output)
    root = root.resolve(strict=True)
    output = output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("--output must be independent of --download-dir (no nested directories)")
    if not root.is_dir() or (output.exists() and not output.is_dir()):
        raise ValueError("Expected input and output directories")
    global_sums = read_checksums(root / GLOBAL_SUMS)
    archives = archive_plan(root, read_checksums(root / "ARCHIVE_SHA256SUMS"))
    metadata = metadata_files(root)
    members = set().union(*(archive["members"] for archive in archives))
    if members.intersection(metadata):
        raise ValueError("Archive members collide with downloaded metadata")
    planned = members | set(metadata)
    if set(global_sums) - planned or members - set(global_sums):
        raise ValueError("Global SHA256SUMS has unavailable files or omits archive members")
    metadata_hashes = {name: file_hash(path) for name, path in metadata.items()}
    for name in set(metadata) & set(global_sums):
        if metadata_hashes[name] != global_sums[name]:
            raise ValueError(f"Downloaded metadata checksum mismatch: {name}")
    expected = {**global_sums, **metadata_hashes}
    check_destinations(output, planned, expected)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, path in metadata.items():
        with path.open("rb") as stream:
            written += copy_verified(stream, output, name, path.stat().st_size, expected[name])
    for archive in archives:
        current = archive["path"].stat()
        previous = archive["stat"]
        if (current.st_size, current.st_mtime_ns, current.st_ino) != (
            previous.st_size,
            previous.st_mtime_ns,
            previous.st_ino,
        ):
            raise ValueError(f"Archive changed after verification: {archive['path']}")
        seen: set[str] = set()
        with tarfile.open(archive["path"], mode="r:", ignore_zeros=True) as stream:
            for member in stream:
                name = validate_member(member, archive["members"], seen)
                content = stream.extractfile(member)
                if content is None:
                    raise ValueError(f"Cannot read archive member: {name}")
                with content:
                    written += copy_verified(content, output, name, member.size, expected[name])
        if seen != archive["members"]:
            raise ValueError("Archive changed during extraction")
        LOGGER.info("Unpacked %s", archive["path"].name)
    for name, digest in global_sums.items():
        if file_hash(output / name) != digest:
            raise ValueError(f"Final expanded-file checksum mismatch: {name}")
    unlisted = sorted(set(metadata) - set(global_sums))
    report = {
        "status": "pass",
        "archives": len(archives),
        "archive_members": len(members),
        "written_files": written,
        "identical_existing_files": len(planned) - written,
        "global_checksum_verified_files": len(global_sums),
        "metadata_not_covered_by_global_checksum_list": unlisted,
        "note": "Unlisted metadata (including the checksum list itself, when omitted) was copied "
        "with source/destination SHA256 equality; it has no entry in the global list.",
    }
    LOGGER.info("Complete: %s", json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and unpack an ECHO-G dataset with the Python standard library.",
        epilog="Use hf download --local-dir first; cached snapshot symlinks are rejected. "
        "checksums/SHA256SUMS is required; its own entry may be omitted. "
        "Existing identical files are accepted; different files are never overwritten.",
    )
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        unpack(args.download_dir, args.output)
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as exc:
        LOGGER.error("Unpack failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
