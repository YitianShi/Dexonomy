#!/usr/bin/env python3
"""Import all 219 official ACRONYM test layouts using the existing 937 MuJoCo assets."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import io
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.prepare_dexgraspnet2_objects import (
    ARCHIVES, RELEASE_OBJECT_COUNT, RELEASE_SCENE_COUNT, REVISION,
)
from scripts.prepare_dexgraspnet2_train_scenes import atomic_json, load_model_configs
from scripts.prepare_graspnet_objects import archive_digest
from dexonomy.op.native_clutter_batch import atomic_npy

SPLITS = {"test_acronym_dense", "test_acronym_random", "test_acronym_loose"}
OFFICIAL_CODE_REVISION = "26ecd76121e3c8218ad53db9840cf34f6b81b076"
OFFICIAL_POSE_REFERENCE = (
    f"https://github.com/PKU-EPIC/DexGraspNet2/blob/{OFFICIAL_CODE_REVISION}/"
    "src/eval/evaluate_dexterous.py#L80-L97"
)
HEIGHT_OFFSET_M = 0.05


def object_poses(annotation):
    """Follow official evaluation: xyzw -> wxyz and subtract 5 cm from world z."""
    if not isinstance(annotation, dict) or not annotation:
        raise ValueError("Expected a nonempty object annotation dictionary")
    poses = {}
    for name, body in annotation.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                or re.fullmatch(r"\d{3}", name) or not isinstance(body, dict)
                or body.get("dataset") != "acronym"):
            raise ValueError(f"Unexpected ACRONYM object: {name}")
        translation = np.asarray(body["rest_pose_trans"], dtype=np.float64).copy()
        xyzw = np.asarray(body["rest_pose_quat"], dtype=np.float64)
        if (translation.shape != (3,) or xyzw.shape != (4,)
                or not np.isfinite(translation).all() or not np.isfinite(xyzw).all()
                or not np.isclose(np.linalg.norm(xyzw), 1., atol=0.002)):
            raise ValueError(f"Invalid pose: {name}")
        translation[2] -= HEIGHT_OFFSET_M
        poses[name] = np.concatenate([translation, xyzw[[3, 0, 1, 2]]])
    return poses


def read_annotations(archive):
    _, size, checksum = ARCHIVES["inventory"]
    archive_digest(archive, expected_bytes=size, expected_sha256=checksum)
    annotations = {}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if (len(parts) != 3 or parts[0] != "acronym_test_scenes"
                    or parts[1] not in SPLITS or not parts[2].endswith(".npz")):
                continue
            name = Path(parts[2]).stem
            split = parts[1].removeprefix("test_acronym_")
            if (not member.isfile() or member.size > 1024 * 1024
                    or not re.fullmatch(rf"scene_{split}_\d+", name) or name in annotations):
                raise ValueError(f"Unexpected scene entry: {member.name}")
            payload = bundle.extractfile(member).read()
            with np.load(io.BytesIO(payload), allow_pickle=True) as data:
                annotation = data["arr_0"].item()
            object_poses(annotation)
            annotations[name] = (member.name, annotation, payload)
    return annotations


def convert_scene(name, source_member, annotation, output, models):
    poses = object_poses(annotation)
    targets = []
    for target in poses:
        path = output / "scene_cfg" / name / f"{target}.npy"
        scene, source_paths = {}, {}
        for object_id, pose in poses.items():
            source_path, model = models[object_id]
            model = copy.deepcopy(model)
            for key, value in model.items():
                if key.endswith("_path") and isinstance(value, str):
                    model[key] = os.path.relpath(value, path.parent)
            model["pose"] = pose.copy()
            scene[object_id] = model
            source_paths[object_id] = os.path.relpath(source_path, REPO_ROOT)
        scene["table"] = {"type": "plane", "pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
                          "size": np.array([0.6, 0.6, 0.01])}
        config = {"scene": scene, "scene_id": f"dexgraspnet2_shapenet_test/{name}/{target}",
                  "task": {"type": "force_closure", "obj_name": target},
                  "clutter": {"generator": "dexgraspnet2_official_acronym", "object_names": list(poses),
                              "source_scene_paths": source_paths, "source_revision": REVISION,
                              "source_archive": "acronym_test_scenes.tar.gz", "source_member": source_member,
                              "coordinate_frame": "table", "unit": "meter",
                              "source_quaternion_order": "xyzw", "height_offset_m": HEIGHT_OFFSET_M,
                              "pose_reference": OFFICIAL_POSE_REFERENCE}}
        atomic_npy(path, config)
        targets.append(str(path.relative_to(output)))
    record = {"scene_name": name, "split": source_member.split("/")[1], "source_member": source_member,
              "object_ids": list(poses), "target_configs": targets,
              "object_poses_wxyz": {key: pose.tolist() for key, pose in poses.items()},
              "coordinate_frame": "table", "unit": "meter", "height_offset_m": HEIGHT_OFFSET_M}
    atomic_json(output / "scenes" / f"{name}.json", record)
    return record


def verify_scene(path):
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg
    env = MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=str(REPO_ROOT / "assets/hand/shadow/right.xml"), freejoint=True),
        scene_cfg=load_scene_cfg(str(path)),
        sim_cfg=MuJoCo_OptCfg(obj_margin=0., hand_margin=0., plane_margin=0.),
        debug_render=False, debug_view=False,
    )
    return {"scene": path.parent.name, "ngeom": env._model.ngeom, "nq": env._model.nq}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=REPO_ROOT / "downloads/dexgraspnet2_acronym_test_scenes.tar.gz")
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "assets/object/dexgraspnet2_shapenet")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets/scene/dexgraspnet2_shapenet_test")
    parser.add_argument("--verify-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.verify_workers < 1:
        parser.error("--verify-workers must be positive")
    args.archive, args.models, args.output = [p.resolve() for p in (args.archive, args.models, args.output)]
    if args.output.is_relative_to(args.models) or args.models.is_relative_to(args.output):
        parser.error("Scene output must be separate from the object assets")
    args.output.mkdir(parents=True, exist_ok=True)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    with (args.output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = {"pid": os.getpid(), "stage": "running", "started_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(args.output / "runtime.json", runtime)
        try:
            annotations = read_annotations(args.archive)
            object_ids = sorted({name for _, annotation, _ in annotations.values() for name in annotation})
            expected = json.loads((args.models / "object_inventory.json").read_text())
            if (len(annotations) != RELEASE_SCENE_COUNT or len(object_ids) != RELEASE_OBJECT_COUNT
                    or object_ids != expected["object_ids"]
                    or {member for member, _, _ in annotations.values()} != set(expected["annotations"])):
                raise ValueError("Test scene/object inventory differs from the prepared 219-scene/937-object release")
            models = load_model_configs(args.models, object_ids)
            records = []
            for name, (member, annotation, payload) in sorted(annotations.items()):
                raw = args.output / "raw" / member
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(payload)
                records.append(convert_scene(name, member, annotation, args.output, models))
            paths = [args.output / record["target_configs"][0] for record in records]
            print(f"Imported {len(records)} scenes; compiling each complete layout in MuJoCo", flush=True)
            with ProcessPoolExecutor(max_workers=args.verify_workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                verification = list(pool.map(verify_scene, paths))
            atomic_json(args.output / "verification.json", {"passed": True, "scenes": verification})
            manifest = {"completed": True, "source_revision": REVISION, "archive": str(args.archive),
                        "archive_sha256": ARCHIVES["inventory"][2], "scene_count": len(records),
                        "object_count": len(object_ids), "object_ids": object_ids,
                        "target_count": sum(len(record["target_configs"]) for record in records),
                        "models": str(args.models), "coordinate_frame": "table", "unit": "meter",
                        "source_quaternion_order": "xyzw", "height_offset_m": HEIGHT_OFFSET_M,
                        "pose_reference": OFFICIAL_POSE_REFERENCE, "mujoco_verified_scenes": len(verification),
                        "scenes": records}
            atomic_json(args.output / "import_manifest.json", manifest)
            runtime.update(stage="completed", finished_at=datetime.now(timezone.utc).isoformat())
            print(f"Ready: {len(records)} scenes, {len(object_ids)} objects, {manifest['target_count']} targets", flush=True)
        except BaseException as exc:
            runtime.update(stage="failed", error=str(exc))
            raise
        finally:
            atomic_json(args.output / "runtime.json", runtime)


if __name__ == "__main__":
    main()
