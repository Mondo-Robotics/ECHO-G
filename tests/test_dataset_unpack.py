# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/unpack_dataset.py"
spec = importlib.util.spec_from_file_location("unpack_dataset", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class UnpackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.download = self.root / "download"
        self.output = self.root / "output"
        (self.download / "data").mkdir(parents=True)
        self.members = [
            ("motion/a.pt", b"motion bytes", tarfile.REGTYPE),
            ("audio/a.wav", b"wav bytes", tarfile.REGTYPE),
        ]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fixture(
        self,
        members: list[tuple[str, bytes, bytes]] | None = None,
        allowed: list[str] | None = None,
        wrong_global: bool = False,
    ) -> None:
        members = self.members if members is None else members
        archive = self.download / "data/part.tar"
        with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
            for name, value, kind in members:
                entry = tarfile.TarInfo(name)
                entry.type = kind
                entry.size = len(value) if kind == tarfile.REGTYPE else 0
                if kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
                    entry.linkname = "../outside"
                tar.addfile(entry, io.BytesIO(value) if kind == tarfile.REGTYPE else None)
        allowed = [name for name, _, _ in members] if allowed is None else allowed
        archive_hash = digest(archive.read_bytes())
        manifest = {
            "schema": "echo-g-dataset-shards-v1",
            "files": len(allowed),
            "shards": [
                {
                    "path": "data/part.tar",
                    "sha256": archive_hash,
                    "bytes": archive.stat().st_size,
                    "files": len(allowed),
                    "member_paths": allowed,
                }
            ],
        }
        (self.download / "data_archives.json").write_text(json.dumps(manifest))
        (self.download / "ARCHIVE_SHA256SUMS").write_text(archive_hash + "  data/part.tar\n")
        (self.download / "README.md").write_bytes(b"dataset documentation")
        (self.download / "checksums").mkdir(exist_ok=True)
        sums = []
        values = {name: value for name, value, _ in members}
        for name in allowed:
            value = values.get(name, b"missing")
            sums.append(("0" * 64 if wrong_global else digest(value)) + "  " + name + "\n")
        sums.append(digest(b"dataset documentation") + "  README.md\n")
        (self.download / "checksums/SHA256SUMS").write_text("".join(sums))

    def test_extract_verify_and_idempotent_resume(self) -> None:
        self.fixture()
        (self.download / ".cache").mkdir()
        (self.download / ".cache" / "ignored").write_bytes(b"cache")
        (self.download / ".git").mkdir()
        first = module.unpack(self.download, self.output)
        self.assertEqual(first["archive_members"], 2)
        self.assertEqual(first["global_checksum_verified_files"], 3)
        self.assertIn("checksums/SHA256SUMS", first["metadata_not_covered_by_global_checksum_list"])
        self.assertFalse((self.output / "data").exists())
        self.assertFalse((self.output / ".cache").exists())
        self.assertEqual((self.output / "motion/a.pt").read_bytes(), b"motion bytes")
        second = module.unpack(self.download, self.output)
        self.assertEqual(second["written_files"], 0)

    def test_corrupt_archive_hash(self) -> None:
        self.fixture()
        path = self.download / "data/part.tar"
        data = bytearray(path.read_bytes())
        data[600] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            module.unpack(self.download, self.output)
        self.assertFalse(self.output.exists())

    def test_unsafe_paths(self) -> None:
        for name in ["../escaped", "/absolute", "dir/../escaped", "C:/escaped", "dir//file"]:
            with self.subTest(name=name):
                self.fixture(members=[(name, b"x", tarfile.REGTYPE)], allowed=["motion/a.pt"])
                with self.assertRaises(ValueError):
                    module.unpack(self.download, self.output)
                self.assertFalse(self.output.exists())

    def test_links_and_special_members(self) -> None:
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.DIRTYPE]:
            with self.subTest(kind=kind):
                self.fixture(members=[("motion/a.pt", b"", kind)])
                with self.assertRaisesRegex(ValueError, "ordinary file"):
                    module.unpack(self.download, self.output)
                self.assertFalse(self.output.exists())

    def test_repeated_tar_member(self) -> None:
        self.fixture(members=[self.members[0], self.members[0]], allowed=["motion/a.pt"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.unpack(self.download, self.output)

    def test_unlisted_and_missing_tar_members(self) -> None:
        for allowed in [["motion/a.pt"], ["motion/a.pt", "audio/a.wav", "extra/file"]]:
            with self.subTest(allowed=allowed):
                self.fixture(allowed=allowed)
                with self.assertRaises(ValueError):
                    module.unpack(self.download, self.output)
                self.assertFalse(self.output.exists())

    def test_duplicate_shard_member(self) -> None:
        self.fixture()
        path = self.download / "data_archives.json"
        d = json.loads(path.read_text())
        other = copy.deepcopy(d["shards"][0])
        other["path"] = "data/second.tar"
        d["shards"].append(other)
        path.write_text(json.dumps(d))
        with (self.download / "ARCHIVE_SHA256SUMS").open("a") as f:
            f.write(other["sha256"] + "  data/second.tar\n")
        with self.assertRaisesRegex(ValueError, "Repeated member"):
            module.unpack(self.download, self.output)

    def test_existing_different_content_is_preserved(self) -> None:
        self.fixture()
        target = self.output / "motion/a.pt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user data")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            module.unpack(self.download, self.output)
        self.assertEqual(target.read_bytes(), b"user data")

    def test_target_symlink_is_rejected(self) -> None:
        self.fixture()
        outside = self.root / "outside"
        outside.mkdir()
        self.output.mkdir()
        (self.output / "motion").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "links"):
            module.unpack(self.download, self.output)
        self.assertEqual(list(outside.iterdir()), [])

    def test_source_metadata_symlink_is_rejected(self) -> None:
        self.fixture()
        (self.download / "README.md").unlink()
        (self.root / "outside").write_bytes(b"dataset documentation")
        (self.download / "README.md").symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "links"):
            module.unpack(self.download, self.output)

    def test_global_content_hash_is_enforced(self) -> None:
        self.fixture(wrong_global=True)
        with self.assertRaisesRegex(ValueError, "content checksum"):
            module.unpack(self.download, self.output)
        self.assertFalse((self.output / "motion/a.pt").exists())

    def test_metadata_hash_is_enforced(self) -> None:
        self.fixture()
        (self.download / "README.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "metadata checksum"):
            module.unpack(self.download, self.output)
        self.assertFalse(self.output.exists())

    def test_checksum_inventory_disagreement(self) -> None:
        self.fixture()
        (self.download / "ARCHIVE_SHA256SUMS").write_text("0" * 64 + "  data/part.tar\n")
        with self.assertRaisesRegex(ValueError, "inventories disagree"):
            module.unpack(self.download, self.output)

    def test_file_directory_collision(self) -> None:
        self.fixture(members=[("a", b"x", tarfile.REGTYPE), ("a/b", b"y", tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, "collision"):
            module.unpack(self.download, self.output)

    def test_nested_output_rejected(self) -> None:
        self.fixture()
        with self.assertRaisesRegex(ValueError, "independent"):
            module.unpack(self.download, self.download / "output")

    def test_global_list_may_not_omit_archive_members(self) -> None:
        self.fixture()
        sums = self.download / "checksums/SHA256SUMS"
        sums.write_text(sums.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "omits archive"):
            module.unpack(self.download, self.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
