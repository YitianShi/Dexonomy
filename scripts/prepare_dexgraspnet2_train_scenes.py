#!/usr/bin/env python3
"""Download official DexGraspNet 2.0 training scenes, retaining geometry metadata only."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION = "b7dbc0ce452d5ab2eda66016ba22d6a1e25e71a4"
SOURCE = "https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0"
# Sizes and SHA256 are the official LFS records at the pinned revision.
ARCHIVES = (
    (6715047936, "557ebc64b0b5beb52a94db0135ff4f824daa673dad96881a30d76148b7d02a18"),
    (15577788702, "2f3c16ad1ea30a942c2a5439e172e839ae1d3d4d043759577e89bf6bcb3e704d"),
    (15612651176, "3244547486ff33c03d3b6ac7b94bdb498a66118f2b647d837868fc560fe40bd0"),
    (16170722874, "c9662632f8285d628192d90240022e62d07aa9bf250e17b0a51ab696831467c3"),
    (16639204712, "6f76a7d3e297fdb765cb750690946c041617936572cace9cd6221e7a4d7dfcbf"),
    (15142850091, "9339d86374aa877fcb2497cde837d50cac7a72a140ef9948227dddee7f5e6a74"),
    (15383641439, "ad666582c30092f5972398915a6b35a05b8cbabdc3b40954b8b7745cb5d76ba7"),
    (15946793524, "e61e0572e7b865ef562eb703a7093b3db6e5841631dda69249a6d017b9090b80"),
    (7424005359, "ecfb24757959a6e812dd26550afd26e5753d7c0600e0784b00097cf81ac46b4b"),
)
TRAIN_IDS = frozenset(range(100)) | frozenset(range(1000, 8500))
RAW_FILES = frozenset({"annotations/0000.xml", "camera_poses.npy", "cam0_wrt_table.npy"})
REPORT_LOCK = threading.Lock()


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def report(output: Path, part: int, **state):
    with REPORT_LOCK:
        path = output / "progress" / f"part_{part}.json"
        previous = json.loads(path.read_text()) if path.is_file() else {}
        previous.update(state)
        previous.update(part=part, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(path, previous)


def download(part: int, archive_dir: Path, output: Path, enabled: bool) -> Path:
    expected_bytes, expected_hash = ARCHIVES[part]
    name = f"scenes_{part}.tar.gz"
    path = archive_dir / name
    partial = path.with_name(path.name + ".part")
    archive_dir.mkdir(parents=True, exist_ok=True)
    existing = path if path.is_file() else partial
    offset = existing.stat().st_size if existing.exists() else 0
    if offset > expected_bytes:
        raise ValueError(f"Oversized archive: {existing}")
    digest = hashlib.sha256()
    if offset:
        report(output, part, stage="verifying" if path.is_file() else "resuming", bytes=offset,
               total_bytes=expected_bytes)
        with existing.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    if path.is_file() and offset != expected_bytes:
        raise ValueError(f"Incomplete final archive: {path}")
    if offset < expected_bytes and not enabled:
        raise FileNotFoundError(f"Missing {path}; use --download")
    failures = 0
    while offset < expected_bytes:
        try:
            # A distinct query avoids a cached redirect for an earlier Range request.
            url = f"{SOURCE}/resolve/{REVISION}/{name}?download=true&resume={offset}"
            with requests.get(url, headers={"Range": f"bytes={offset}-"}, stream=True,
                              timeout=(30, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get(
                        "Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError(f"Server did not honor the download offset for {name}")
                last_report = time.monotonic()
                report(output, part, stage="downloading", bytes=offset, total_bytes=expected_bytes)
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if offset + len(chunk) > expected_bytes:
                            raise ValueError(f"Response exceeded expected archive size: {name}")
                        stream.write(chunk)
                        digest.update(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_report > 10:
                            stream.flush()
                            report(output, part, stage="downloading", bytes=offset,
                                   total_bytes=expected_bytes)
                            print(f"{name}: {offset / expected_bytes:.1%} ({offset / 1e9:.2f} GB)",
                                  flush=True)
                            last_report = time.monotonic()
            if offset != expected_bytes:
                raise requests.ConnectionError(f"Download ended early at {offset} bytes")
        except requests.RequestException as error:
            failures += 1
            report(output, part, stage="retrying", bytes=offset, error=str(error), retries=failures)
            if failures >= 20:
                raise
            time.sleep(min(30, failures * 2))
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"SHA256 differs from official LFS record: {name}")
    if not path.exists():
        partial.replace(path)
    report(output, part, stage="verified", bytes=expected_bytes,
           total_bytes=expected_bytes, sha256=expected_hash)
    return path


def selected_metadata(name: str, camera: str) -> bool:
    parts = PurePosixPath(name).parts
    if len(parts) < 4 or parts[0] != "scenes" or parts[2] != camera:
        return False
    match = re.fullmatch(r"scene_(\d{4})", parts[1])
    return bool(match and int(match[1]) in TRAIN_IDS and "/".join(parts[3:]) in RAW_FILES)


def extract_metadata(archive: Path, output: Path, camera: str, part: int) -> list[str]:
    names: set[str] = set()
    raw_root = output / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    report(output, part, stage="extracting", metadata_scenes=0)
    last_report = time.monotonic()
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not selected_metadata(member.name, camera):
                continue
            if not member.isfile() or member.size > 1024 * 1024:
                raise ValueError(f"Unexpected metadata entry: {member.name}")
            destination = raw_root / member.name
            if not destination.resolve().is_relative_to(raw_root.resolve()):
                raise ValueError(f"Metadata path escapes the destination: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            with bundle.extractfile(member) as source, temporary.open("wb") as target:
                target.write(source.read())
            temporary.replace(destination)
            names.add(PurePosixPath(member.name).parts[1])
            if time.monotonic() - last_report > 10:
                report(output, part, metadata_scenes=len(names))
                last_report = time.monotonic()
    expected_ids = set(range(100)) if part == 0 else {
        i for i in TRAIN_IDS if i // 1000 == part}
    expected_names = {f"scene_{i:04d}" for i in expected_ids}
    if names != expected_names:
        raise ValueError(f"Archive scene inventory differs: missing={sorted(expected_names - names)}, "
                         f"unexpected={sorted(names - expected_names)}")
    for name in names:
        for relative in RAW_FILES:
            if not (raw_root / "scenes" / name / camera / relative).is_file():
                raise ValueError(f"Missing {relative} for {name}/{camera}")
    report(output, part, stage="extracted", metadata_scenes=len(names))
    return sorted(names)


def rigid_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=0.005)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=0.005)):
        raise ValueError("Expected a finite rigid 4x4 transform")
    return value


def object_poses(annotation: Path, table_from_camera: np.ndarray) -> dict[str, np.ndarray]:
    """Official XML quaternions are wxyz; positions are in the selected camera frame."""
    table_from_camera = rigid_transform(table_from_camera)
    root = ET.parse(annotation).getroot()
    poses = {}
    for obj in root.findall("obj"):
        object_id = int(obj.findtext("obj_id"))
        if object_id not in range(88):
            raise ValueError(f"Unknown GraspNet object: {object_id}")
        name = f"{object_id:03d}"
        if name in poses:
            raise ValueError(f"Duplicate object {name}: {annotation}")
        xyz = np.asarray([float(v) for v in obj.findtext("pos_in_world").split()])
        wxyz = np.asarray([float(v) for v in obj.findtext("ori_in_world").split()])
        if (xyz.shape != (3,) or wxyz.shape != (4,) or not np.isfinite(xyz).all()
                or not np.isfinite(wxyz).all() or not np.isclose(np.linalg.norm(wxyz), 1, atol=0.002)):
            raise ValueError(f"Invalid object pose in {annotation}")
        camera_from_object = np.eye(4)
        camera_from_object[:3, 3] = xyz
        camera_from_object[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        table_from_object = table_from_camera @ camera_from_object
        xyzw = Rotation.from_matrix(table_from_object[:3, :3]).as_quat()
        poses[name] = np.concatenate([table_from_object[:3, 3], xyzw[[3, 0, 1, 2]]])
    if not poses:
        raise ValueError(f"Empty scene annotation: {annotation}")
    return poses


def load_model_configs(models: Path, object_ids=None) -> dict[str, tuple[Path, dict]]:
    result = {}
    for name in (object_ids if object_ids is not None else [f"{index:03d}" for index in range(88)]):
        path = models / "scene_cfg" / name / "floating/scale001.npy"
        cfg = np.load(path, allow_pickle=True).item()["scene"][name]
        if not np.allclose(cfg["scale"], np.ones(3)):
            raise ValueError(f"Expected original meter-scale geometry: {name}")
        cfg = copy.deepcopy(cfg)
        for key, value in cfg.items():
            if key.endswith("_path") and isinstance(value, str):
                asset = (path.parent / value).resolve()
                if not asset.is_file():
                    raise FileNotFoundError(asset)
                cfg[key] = str(asset)
        result[name] = (path, cfg)
    return result


def convert_scene(name: str, output: Path, camera: str, part: int, models: dict,
                  *, scene_namespace: str = "dexgraspnet2_train") -> dict:
    metadata = output / "raw/scenes" / name / camera
    camera_poses = np.load(metadata / "camera_poses.npy", allow_pickle=False)
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4) or len(camera_poses) < 1:
        raise ValueError(f"Invalid camera poses: {name}")
    align = rigid_transform(np.load(metadata / "cam0_wrt_table.npy", allow_pickle=False))
    table_from_camera = align @ rigid_transform(camera_poses[0])
    poses = object_poses(metadata / "annotations/0000.xml", table_from_camera)
    targets = []
    for target in poses:
        path = output / "scene_cfg" / name / f"{target}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = {}
        source_paths = {}
        for object_id, pose in poses.items():
            source_path, cfg = models[object_id]
            cfg = copy.deepcopy(cfg)
            for key, value in cfg.items():
                if key.endswith("_path") and isinstance(value, str):
                    cfg[key] = os.path.relpath(value, path.parent)
            cfg["pose"] = pose.copy()
            scene[object_id] = cfg
            source_paths[object_id] = os.path.relpath(source_path, REPO_ROOT)
        scene["table"] = {"type": "plane", "pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
                          "size": np.array([0.6, 0.6, 0.01])}
        cfg = {"scene": scene, "scene_id": f"{scene_namespace}/{name}/{target}",
               "task": {"type": "force_closure", "obj_name": target},
               "clutter": {"generator": "dexgraspnet2_official", "object_names": list(poses),
                           "source_scene_paths": source_paths, "source_revision": REVISION,
                           "source_archive": f"scenes_{part}.tar.gz", "camera": camera,
                           "view": 0, "coordinate_frame": "table", "unit": "meter"}}
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, cfg)
        temporary.replace(path)
        targets.append(str(path.relative_to(output)))
    record = {"scene_name": name, "object_ids": list(poses), "target_configs": targets,
              "object_poses_wxyz": {key: value.tolist() for key, value in poses.items()},
              "table_from_camera_view0": table_from_camera.tolist(), "coordinate_frame": "table",
              "unit": "meter", "camera": camera, "view": 0}
    atomic_json(output / "scenes" / f"{name}.json", record)
    return record


def prepare_part(part: int, args, models: dict) -> dict:
    marker = args.output / "import_parts" / f"part_{part}.json"
    expected_hash = ARCHIVES[part][1]
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if (saved.get("archive_sha256") == expected_hash and saved.get("camera") == args.camera
                and saved.get("models") == str(args.models)
                and all((args.output / p).is_file() for record in saved["scenes"]
                        for p in record["target_configs"])
                and all((args.output / "scenes" / f"{record['scene_name']}.json").is_file()
                        and all((args.output / "raw/scenes" / record["scene_name"] / args.camera / p).is_file()
                                for p in RAW_FILES) for record in saved["scenes"])):
            report(args.output, part, stage="completed", scene_count=len(saved["scenes"]))
            return saved
    archive = download(part, args.archive_dir, args.output, args.download)
    names = extract_metadata(archive, args.output, args.camera, part)
    report(args.output, part, stage="converting", scene_count=0)
    records = []
    for name in names:
        records.append(convert_scene(name, args.output, args.camera, part, models))
        if len(records) % 50 == 0:
            report(args.output, part, scene_count=len(records))
    result = {"part": part, "archive_sha256": expected_hash, "source_revision": REVISION,
              "camera": args.camera, "models": str(args.models), "scenes": records}
    atomic_json(marker, result)
    report(args.output, part, stage="completed", scene_count=len(records),
           target_count=sum(len(r["target_configs"]) for r in records))
    if not args.keep_archives:
        archive.unlink()
    print(f"scenes_{part}: imported {len(records)} training scenes", flush=True)
    return result


def build_manifest(results: list[dict], camera: str) -> dict:
    records = [record for result in sorted(results, key=lambda r: r["part"]) for record in result["scenes"]]
    actual_ids = [int(record["scene_name"][6:]) for record in records]
    object_ids = sorted({name for record in records for name in record["object_ids"]})
    if len(actual_ids) != len(TRAIN_IDS) or set(actual_ids) != TRAIN_IDS:
        raise ValueError("Imported scene inventory differs from the official training split")
    # Derive object counts from the released annotations. The paper reports 60,
    # but the published 100 seed-scene annotations contain 40 distinct object IDs.
    return {"completed": True, "source": SOURCE, "source_revision": REVISION,
            "scene_count": len(records), "object_count": len(object_ids), "object_ids": object_ids,
            "paper_reported_training_objects": 60,
            "target_count": sum(len(record["target_configs"]) for record in records),
            "camera": camera, "view": 0, "coordinate_frame": "table", "unit": "meter",
            "scene_configs": "scene_cfg/scene_*/[0-9][0-9][0-9].npy",
            "retained_metadata": sorted(RAW_FILES), "scenes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent archive downloads/imports")
    parser.add_argument("--camera", choices=("realsense", "kinect"), default="realsense")
    parser.add_argument("--archive-dir", type=Path, default=REPO_ROOT / "downloads/dexgraspnet2_train_scenes")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets/scene/dexgraspnet2_train")
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "assets/object/graspnet_1billion")
    parser.add_argument("--keep-archives", action="store_true", help="Keep compressed RGB/depth archives after import")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output = args.output.resolve()
    args.models = args.models.resolve()
    args.archive_dir = args.archive_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent two launch entries from appending to the same resumable downloads.
    import fcntl
    with (args.output / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This scene preparation is already running")
        atomic_json(args.output / "runtime.json", {"pid": os.getpid(), "stage": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(), "workers": args.workers,
                    "expected_scenes": len(TRAIN_IDS), "download_bytes": sum(a[0] for a in ARCHIVES),
                    "camera": args.camera, "source_revision": REVISION})
        models = load_model_configs(args.models)
        results = []
        errors = {}
        # Tar header parsing is Python work: processes let concurrent imports
        # use separate CPU cores while the existing grasp job keeps running.
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(prepare_part, part, args, models): part for part in range(9)}
            for future in as_completed(futures):
                part = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    report(args.output, part, stage="failed", error=str(error))
                    errors[part] = str(error)
        if errors:
            runtime = json.loads((args.output / "runtime.json").read_text())
            runtime.update(stage="failed", errors=errors)
            atomic_json(args.output / "runtime.json", runtime)
            raise RuntimeError(f"Failed scene archive parts: {sorted(errors)}; rerun to resume")
        manifest = build_manifest(results, args.camera)
        atomic_json(args.output / "import_manifest.json", manifest)
        runtime = json.loads((args.output / "runtime.json").read_text())
        runtime.update(stage="completed", finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / "runtime.json", runtime)
        print(f"Ready: {manifest['scene_count']} official training scenes, "
              f"{manifest['object_count']} objects", flush=True)


if __name__ == "__main__":
    main()
