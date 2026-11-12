#!/usr/bin/env python3
"""Download all released object models or prepare the annotated ShapeNet subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import io
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.prepare_data import validate_member
from scripts.prepare_graspnet_objects import (
    ARCHIVE_BYTES, ARCHIVE_SHA256, OBJECT_IDS, archive_digest, convert_object,
    download_archive, link_transforms, origin_transform,
)

REVISION = "b7dbc0ce452d5ab2eda66016ba22d6a1e25e71a4"
BASE_URL = f"https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/{REVISION}"
ARCHIVES = {
    "acronym": ("acronym.tar.gz", 642645887,
                "5242d5e372cc008fec140a4bd05bf6e0abdbc16af532f59e51d8934e8fa969ec"),
    "inventory": ("acronym_test_scenes.tar.gz", 532596421,
                  "e24feda731f430fa21edc63e74f43e9fa8d2106e831633f68974848f102aee00"),
}
CONVEX_SETTINGS = {"maxConvexHulls": 64, "resolution": 400000,
                   "minimumVolumePercentErrorAllowed": 1.0, "asyncACD": False}
RELEASE_OBJECT_COUNT = 937
RELEASE_SCENE_COUNT = 219
PLANAR_THICKNESS_M = 1e-6
MODEL_FILES = {"scaled.obj", "simplified.obj", "collision.urdf"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read_inventory(archive):
    """Only asset IDs are retained; clutter poses never enter the grasp pipeline."""
    objects, scenes = set(), {}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 3 or
                    parts[0] != "acronym_test_scenes" or
                    parts[1] not in {"test_acronym_dense", "test_acronym_random", "test_acronym_loose"} or
                    not parts[2].endswith(".npz")):
                continue
            # These pickled annotations are read only after checking the official archive SHA256.
            with np.load(io.BytesIO(bundle.extractfile(member).read()), allow_pickle=True) as data:
                annotation = data["arr_0"].item()
            for object_id, body in annotation.items():
                if (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", object_id) or
                        re.fullmatch(r"\d{3}", object_id) or body.get("dataset") != "acronym"):
                    raise ValueError(f"Unexpected ShapeNet asset in release inventory: {object_id}")
                objects.add(object_id)
            scenes[member.name] = sorted(annotation)
    if not objects:
        raise ValueError("Official test archive contains no ShapeNet object annotations")
    named = sorted(object_id for object_id in objects if not re.fullmatch(r"[0-9a-f]{20,32}", object_id))
    return {"object_ids": sorted(objects), "object_count": len(objects), "named_object_ids": named,
            "annotation_count": len(scenes), "annotations": scenes}


def read_model_inventory(archive):
    """Enumerate complete physical models independently of the scene annotations."""
    found = {}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 5 or
                    parts[:3] != ("acronym", "meshes", "models")):
                continue
            object_id = parts[3]
            if object_id in {".", ".."} or "\\" in object_id or "\x00" in object_id:
                raise ValueError(f"Unexpected model ID: {object_id}")
            files = found.setdefault(object_id, set())
            if member.size > 0:
                files.add(parts[4])
    complete = sorted(key for key, files in found.items() if MODEL_FILES <= files)
    incomplete = {key: sorted(MODEL_FILES - files) for key, files in sorted(found.items())
                  if not MODEL_FILES <= files}
    if not complete:
        raise ValueError("Official model archive contains no complete physical geometry")
    return {"object_ids": complete, "object_count": len(complete),
            "archive_model_directory_count": len(found),
            "incomplete_geometry_count": len(incomplete), "incomplete_geometry": incomplete}


def extract_full_model_archive(archive, destination, archive_root):
    """Retain every official model file, including incomplete model directories."""
    files = total_bytes = 0
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != archive_root:
                raise ValueError(f"Unexpected model archive root: {member.name}")
            validate_member(member, destination)
            path = destination.joinpath(*parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported model archive member: {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with bundle.extractfile(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if temporary.stat().st_size != member.size:
                raise ValueError(f"Incomplete extracted model file: {member.name}")
            temporary.replace(path)
            files += 1
            total_bytes += member.size
    return {"files": files, "unpacked_bytes": total_bytes}


def build_benchmark_inventory(annotations, model_catalog):
    """Count only IDs established by the released 2.0 benchmark annotations."""
    annotated = set(annotations["object_ids"])
    confirmed = [item for item in model_catalog
                 if item["dataset"] == "graspnet" or
                 (item["dataset"] == "acronym" and item["object_id"] in annotated)]
    return {
        "dataset": "DexGraspNet 2.0", "revision": REVISION,
        "status": "complete_id_mapping" if len(confirmed) == 1319 else "incomplete_id_mapping",
        "paper_benchmark_object_count": 1319, "paper_shapenet_object_count": 1231,
        "confirmed_object_count": len(confirmed),
        "confirmed_acronym_object_count": len(annotated),
        "confirmed_graspnet_object_count": sum(item["dataset"] == "graspnet" for item in confirmed),
        "unresolved_benchmark_id_count": max(0, 1319 - len(confirmed)),
        "released_annotation_count": annotations["annotation_count"],
        "objects": confirmed,
        "paper_source": "https://arxiv.org/html/2410.23004v1#A3.SS1",
        "selection": "all 88 official GraspNet models plus unique IDs present in released ShapeNet test annotations",
        "note": "Unannotated models in the full ACRONYM pool are excluded; unresolved IDs do not establish that mesh files are missing.",
    }


def prepare_full_release(args):
    """Download/extract the full release without changing an existing grasp library."""
    output = (args.output or REPO_ROOT / "assets/object/dexgraspnet2_full").resolve()
    archives = dict(ARCHIVES, meshdata=("meshdata.tar.gz", ARCHIVE_BYTES, ARCHIVE_SHA256))
    paths = {}
    verified = {}
    for key, (filename, size, sha) in archives.items():
        path = args.downloads.resolve() / filename
        if args.download:
            download_archive(path, url=f"{BASE_URL}/{filename}?download=true", expected_bytes=size)
        archive_digest(path, expected_bytes=size, expected_sha256=sha)
        paths[key] = path
        verified[key] = {"filename": filename, "size_bytes": size, "sha256": sha,
                         "local_path": str(path), "source_url": f"{BASE_URL}/{filename}"}
    inventory = read_model_inventory(paths["acronym"])
    annotations = read_inventory(paths["inventory"])
    if not set(annotations["object_ids"]) <= set(inventory["object_ids"]):
        raise ValueError("Annotated objects are absent from the complete geometry inventory")
    extraction = {"acronym": extract_full_model_archive(paths["acronym"], output, "acronym"),
                  "graspnet": extract_full_model_archive(paths["meshdata"], output, "meshdata")}
    graspnet_ids = sorted(path.name for path in (output / "meshdata").iterdir() if path.is_dir())
    if graspnet_ids != list(OBJECT_IDS):
        raise ValueError("Full model package does not contain exactly the 88 official GraspNet IDs")
    for object_id in graspnet_ids:
        for name in ("simplified.obj", "coacd/coacd.urdf"):
            path = output / "meshdata" / object_id / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing GraspNet generation geometry: {path}")
    for object_id in inventory["object_ids"]:
        for name in MODEL_FILES:
            path = output / "acronym/meshes/models" / object_id / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing physical ShapeNet geometry: {path}")
    combined = [{"dataset": "acronym", "object_id": object_id,
                 "model_path": str(output / "acronym/meshes/models" / object_id)}
                for object_id in inventory["object_ids"]]
    combined.extend({"dataset": "graspnet", "object_id": object_id,
                     "model_path": str(output / "meshdata" / object_id)} for object_id in graspnet_ids)
    if len(combined) < 1300:
        raise ValueError(f"Only {len(combined)} complete object models; requested at least 1300")
    write_json(output / "incomplete_geometry.json", inventory["incomplete_geometry"])
    write_json(output / "object_inventory.json", {
        "object_count": len(combined), "acronym_object_count": inventory["object_count"],
        "graspnet_object_count": len(graspnet_ids), "objects": combined,
        "selection": "all released models with complete physical geometry, not the paper's benchmark subset",
    })
    write_json(output / "released_test_inventory.json", annotations)
    benchmark = build_benchmark_inventory(annotations, combined)
    write_json(output / "benchmark_inventory.json", benchmark)
    write_json(output / "import_manifest.json", {
        "status": "raw_complete", "revision": REVISION, "archives": verified,
        "object_count": len(combined), "acronym_object_count": inventory["object_count"],
        "graspnet_object_count": len(graspnet_ids),
        "acronym_archive_model_directories": inventory["archive_model_directory_count"],
        "incomplete_geometry_count": inventory["incomplete_geometry_count"],
        "released_test_object_count": annotations["object_count"],
        "paper_benchmark_object_count": 1319, "paper_benchmark_shapenet_count": 1231,
        "confirmed_benchmark_object_count": benchmark["confirmed_object_count"],
        "unresolved_benchmark_id_count": benchmark["unresolved_benchmark_id_count"],
        "selection": "full official model archives; all files retained, complete physical models indexed separately",
        "collision_policy": "official raw assets retained; full ACRONYM models require MuJoCo conversion before grasp generation",
        "extraction": extraction,
    })
    print(json.dumps({"output": str(output), "complete_objects": len(combined),
                      "acronym_objects": inventory["object_count"], "graspnet_objects": len(graspnet_ids),
                      "incomplete_model_directories": inventory["incomplete_geometry_count"]}, indent=2), flush=True)


def extract_models(archive, destination, object_ids):
    selected = set(object_ids)
    retained = {"scaled.obj", "simplified.obj", "collision.urdf"}
    found = {object_id: set() for object_id in object_ids}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if (len(parts) != 5 or parts[:3] != ("acronym", "meshes", "models") or
                    parts[3] not in selected or parts[4] not in retained):
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported model archive member: {member.name}")
            validate_member(member, destination)
            path = destination / parts[3] / parts[4]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with bundle.extractfile(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(path)
            found[parts[3]].add(parts[4])
    missing = {key: sorted(retained - files) for key, files in found.items() if files != retained}
    if missing:
        raise ValueError(f"Selected models are missing official geometry: {missing}")


def physical_surface(source):
    """Use the meter-scale collision surface; simplified.obj is unscaled ShapeNet geometry."""
    # The release files start with a blank line before their XML declaration.
    robot = ET.fromstring((source / "collision.urdf").read_bytes().lstrip())
    transforms = link_transforms(robot)
    collisions = [(link, collision) for link in robot.findall("link")
                  for collision in link.findall("collision")]
    if len(collisions) != 1:
        raise ValueError(f"Expected one official scaled surface: {source.name}")
    link, collision = collisions[0]
    mesh = collision.find("geometry/mesh")
    if mesh is None or mesh.get("filename") != "scaled.obj":
        raise ValueError(f"Unexpected official collision surface: {source.name}")
    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    transform = transforms[link.get("name")] @ origin_transform(collision)
    if scale.shape != (3,) or not np.array_equal(scale, np.ones(3)) or not np.allclose(transform, np.eye(4)):
        raise ValueError(f"Unexpected official surface transform: {source.name}")
    surface = trimesh.load(source / "scaled.obj", force="mesh", process=False, skip_materials=True)
    if not len(surface.faces) or not np.isfinite(surface.vertices).all() or np.any(surface.extents <= 0):
        raise ValueError(f"Invalid meter-scale surface: {source.name}")
    return surface


def mujoco_convex_part(vertices, faces):
    """Retain zero-thickness V-HACD fragments with micrometer collision thickness."""
    vertices = np.asarray(vertices, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not len(mesh.faces) or not np.isfinite(vertices).all():
        raise ValueError("Invalid V-HACD convex part")
    _, singular, directions = np.linalg.svd(vertices - vertices.mean(0), full_matrices=True)
    tolerance = max(singular[0] * 1e-12, 1e-12)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 3:
        return mesh, False
    expanded = vertices
    for normal in directions[rank:]:
        expanded = np.concatenate((expanded + normal * PLANAR_THICKNESS_M / 2,
                                   expanded - normal * PLANAR_THICKNESS_M / 2))
    return trimesh.convex.convex_hull(expanded), True


def prepare_object(object_id, output):
    raw = output / "raw_source" / object_id
    source_hash = hashlib.sha256((raw / "scaled.obj").read_bytes()).hexdigest()
    fingerprint = {"source_mesh_sha256": source_hash, "convex_settings": CONVEX_SETTINGS,
                   "vhacdx_version": version("vhacdx"), "trimesh_version": version("trimesh"),
                   "face_encoding": "VTK polydata via trimesh.convex_decomposition"}
    marker = output / "processed_data" / object_id / "import.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("fingerprint") == fingerprint and Path(saved["scene_path"]).is_file():
            return saved
    surface = physical_surface(raw)
    # The binding expects flattened VTK faces [3, i, j, k, ...], not an (N, 3)
    # triangle array. Trimesh supplies this encoding through its public interface.
    parts = trimesh.decomposition.convex_decomposition(surface, **CONVEX_SETTINGS)
    if not parts:
        raise ValueError(f"Convex decomposition produced no geometry: {object_id}")
    conversion = output / "convex_source" / object_id
    convex = conversion / "coacd"
    if convex.exists():
        shutil.rmtree(convex)
    convex.mkdir(parents=True, exist_ok=True)
    # Copy the physical mesh byte-for-byte into the existing converter's surface slot.
    shutil.copyfile(raw / "scaled.obj", conversion / "simplified.obj")
    robot = ET.Element("robot", name=object_id)
    link = ET.SubElement(robot, "link", name="object")
    regularized = []
    for index, part in enumerate(parts):
        mesh, expanded = mujoco_convex_part(part["vertices"], part["faces"])
        if expanded:
            regularized.append(index)
        name = f"convex_{index:03d}.obj"
        mesh.export(convex / name, digits=17)
        geometry = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
        ET.SubElement(geometry, "mesh", filename=name)
    ET.indent(robot)
    ET.ElementTree(robot).write(convex / "coacd.urdf", encoding="unicode")
    previous_urdf = output / "processed_data" / object_id / "urdf"
    if previous_urdf.exists():
        shutil.rmtree(previous_urdf)
    record = convert_object(object_id, output / "convex_source", output,
                            allowed_ids=(object_id,), model_prefix="dexgraspnet2")
    record.update(fingerprint=fingerprint, source_mesh=str(raw / "scaled.obj"),
                  collision_algorithm="V-HACD", scale_policy="official scaled.obj in meters, scale [1,1,1]",
                  regularized_convex_meshes=regularized, planar_collision_thickness_m=PLANAR_THICKNESS_M)
    write_json(marker, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=REPO_ROOT / "downloads")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--full-release", action="store_true",
                        help="Extract/index all official model files instead of the annotated 937-object subset.")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.full_release:
        prepare_full_release(args)
        return
    output = (args.output or REPO_ROOT / "assets/object/dexgraspnet2_shapenet").resolve()
    archives = {}
    for key, (filename, size, sha) in ARCHIVES.items():
        path = args.downloads.resolve() / f"dexgraspnet2_{filename}"
        if args.download:
            download_archive(path, url=f"{BASE_URL}/{filename}?download=true", expected_bytes=size)
        archive_digest(path, expected_bytes=size, expected_sha256=sha)
        archives[key] = path
    inventory = read_inventory(archives["inventory"])
    if (inventory["object_count"], inventory["annotation_count"]) != (RELEASE_OBJECT_COUNT, RELEASE_SCENE_COUNT):
        raise ValueError("Inventory differs from the pinned official release")
    write_json(output / "object_inventory.json", inventory)
    write_json(output / "import_manifest.json", {"status": "preparing", "object_count": inventory["object_count"]})
    extract_models(archives["acronym"], output / "raw_source", inventory["object_ids"])
    print(f"Prepared official inventory: {inventory['object_count']} ShapeNet objects; original meter-scale geometry", flush=True)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    records, errors = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(prepare_object, object_id, output): object_id
                   for object_id in inventory["object_ids"]}
        for future in as_completed(pending):
            object_id = pending[future]
            try:
                records[object_id] = future.result()
            except Exception as error:
                errors[object_id] = str(error)
            done = len(records) + len(errors)
            write_json(output / "prepare_progress.json", {"completed": len(records),
                       "total": inventory["object_count"], "errors": errors})
            if done % 25 == 0 or done == inventory["object_count"]:
                print(f"MuJoCo assets: {done}/{inventory['object_count']}; failures: {len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"Failed to prepare {len(errors)} objects; see {output / 'prepare_progress.json'}")
    write_json(output / "valid_split/all.json", inventory["object_ids"])
    write_json(output / "import_manifest.json", {
        "status": "complete",
        "source_url": "https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0",
        "revision": REVISION, "archives": {key: {"filename": filename, "sha256": sha}
                    for key, (filename, _, sha) in ARCHIVES.items()},
        "object_count": len(records), "object_ids": inventory["object_ids"],
        "selection": "unique ShapeNet IDs in released test_acronym_dense/random/loose annotations; excludes GraspNet 000-087",
        "paper_object_count": 1231, "released_test_object_count": RELEASE_OBJECT_COUNT,
        "scale_policy": "use official scaled.obj in meters without normalization, one scale [1,1,1]",
        "collision_policy": "V-HACD convex decomposition of scaled.obj; validate every MuJoCo model",
        "planar_collision_thickness_m": PLANAR_THICKNESS_M,
        "objects": [records[object_id] for object_id in inventory["object_ids"]],
    })
    print(f"Prepared {len(records)} DexGraspNet 2.0 ShapeNet objects: {output}", flush=True)


if __name__ == "__main__":
    main()
