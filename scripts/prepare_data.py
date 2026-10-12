#!/usr/bin/env python3
"""Safely unpack the Dexonomy archives into the paths expected by the README."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "DGN_5k_processed.tar.gz": "asset",
    "DGN_5k_vision.tar.gz": "asset",
    "Dexonomy_GRASP_shadow.tar.gz": "output",
    "allegro_fingertip.tar.gz": "output",
    "dexonomy_learning_ckpt.tar.gz": "output",
    "IsaacGym_Preview_4_Package.tar.gz": "output",
}
GROUPS = {
    "processed": ["DGN_5k_processed.tar.gz"],
    "vision": ["DGN_5k_vision.tar.gz"],
    "grasps": ["Dexonomy_GRASP_shadow.tar.gz"],
    "core": [
        "DGN_5k_processed.tar.gz",
        "DGN_5k_vision.tar.gz",
        "Dexonomy_GRASP_shadow.tar.gz",
    ],
    "all": list(ARCHIVES),
}


def validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    destination = destination.resolve()
    member_path = (destination / member.name).resolve()
    if os.path.commonpath((str(destination), str(member_path))) != str(destination):
        raise ValueError(f"Unsafe path in archive: {member.name}")
    if member.issym() or member.islnk():
        link_path = (member_path.parent / member.linkname).resolve()
        if os.path.commonpath((str(destination), str(link_path))) != str(destination):
            raise ValueError(f"Unsafe link in archive: {member.name} -> {member.linkname}")


def extract_archive(archive: Path, destination: Path, force: bool, marker_root: Path) -> bool:
    marker = marker_root / f"{archive.name}.done"
    if marker.is_file() and not force:
        print(f"[skip] {archive.name} (marker exists)", flush=True)
        return False

    destination.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive.name} -> {destination}", flush=True)
    files = 0
    unpacked_bytes = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            validate_member(member, destination)
            bundle.extract(member, destination)
            if member.isfile():
                files += 1
                unpacked_bytes += member.size
                if files % 25_000 == 0:
                    print(f"  {files:,} files, {unpacked_bytes / 1024**3:.1f} GiB unpacked", flush=True)

    marker_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive": str(archive.resolve()),
        "destination": str(destination.resolve()),
        "files": files,
        "unpacked_bytes": unpacked_bytes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(marker)
    print(f"[done] {archive.name}: {files:,} files, {unpacked_bytes / 1024**3:.1f} GiB", flush=True)
    return True


def check_space(archives: list[Path], target: Path) -> None:
    compressed = sum(path.stat().st_size for path in archives)
    free = shutil.disk_usage(target).free
    estimate = compressed * 4
    print(
        f"Disk check: {compressed / 1024**3:.1f} GiB compressed, "
        f"approximately {estimate / 1024**3:.1f} GiB required, {free / 1024**3:.1f} GiB free",
        flush=True,
    )
    if free < estimate:
        raise OSError("Not enough free disk space for the conservative 4x extraction estimate.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, default="core", help="Archive group to unpack (default: core).")
    parser.add_argument("--archive-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "extracted")
    parser.add_argument("--force", action="store_true", help="Extract again even when a completion marker exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.resolve()
    asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve()
    selected = [archive_dir / name for name in GROUPS[args.group]]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        names = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing archives:\n{names}")

    asset_target = asset_root / "object"
    marker_root = archive_dir / ".dexonomy_extracted"
    asset_target.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    check_space(selected, archive_dir)

    for archive in selected:
        destination = asset_target if ARCHIVES[archive.name] == "asset" else output_root
        extract_archive(archive, destination, args.force, marker_root)

if __name__ == "__main__":
    main()
