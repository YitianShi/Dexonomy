#!/usr/bin/env python3
"""Import the 88 official GraspNet objects using DexGraspNet 2.0's convex meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import requests
from scipy.spatial.transform import Rotation
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.prepare_data import validate_member

ARCHIVE_URL = "https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/resolve/main/meshdata.tar.gz?download=true"
ARCHIVE_BYTES = 4617887647
ARCHIVE_SHA256 = "55a13b63f3838881b25ec72187d922454ff5d69ce019816f7f29510b14c13f0f"
OBJECT_IDS = tuple(f"{i:03d}" for i in range(88))


def download_archive(path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > ARCHIVE_BYTES:
        raise ValueError(f"Oversized partial download: {partial}")
    if offset < ARCHIVE_BYTES:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(ARCHIVE_URL, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if offset and (response.status_code != 206 or not response.headers.get(
                    "Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Server did not honor the download resume offset")
            last_report = time.monotonic()
            with partial.open("ab" if offset else "wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - last_report > 20:
                        print(f"Download: {offset / ARCHIVE_BYTES:.1%} ({offset / 1e9:.2f} GB)", flush=True)
                        last_report = time.monotonic()
    if partial.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"Incomplete archive: {partial}")
    partial.replace(path)


def archive_digest(path):
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"Archive size differs from the official release: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 differs from the official Hugging Face LFS checksum")
    return digest.hexdigest()


def extract_geometry(archive, destination, digest):
    marker = destination / "extraction.json"
    if marker.is_file() and json.loads(marker.read_text()).get("archive_sha256") == digest:
        return destination / "meshdata"
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    objects = set()
    files = 0
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if (len(parts) < 3 or parts[0] != "meshdata" or parts[1] not in OBJECT_IDS or
                    (parts[2] != "coacd" and parts[2:] != ("simplified.obj",))):
                continue
            validate_member(member, temporary)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported geometry archive member: {member.name}")
            bundle.extract(member, temporary, filter="data")
            objects.add(parts[1])
            files += member.isfile()
    missing = set(OBJECT_IDS) - objects
    if missing:
        raise ValueError(f"Official model archive is missing objects: {sorted(missing)}")
    (temporary / "extraction.json").write_text(json.dumps({
        "archive_sha256": digest, "object_count": len(objects), "files": files,
        "retained": ["coacd/**", "simplified.obj"],
    }, indent=2) + "\n")
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    print(f"Extracted generation geometry for all {len(objects)} objects ({files} files)", flush=True)
    return destination / "meshdata"


def origin_transform(element):
    result = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def link_transforms(robot):
    links = {link.get("name") for link in robot.findall("link")}
    parents = {}
    for joint in robot.findall("joint"):
        if joint.get("type") != "fixed":
            raise ValueError("GraspNet objects must be rigid: found a non-fixed joint")
        child, parent = joint.find("child").get("link"), joint.find("parent").get("link")
        if child in parents or child not in links or parent not in links:
            raise ValueError("Invalid URDF link tree")
        parents[child] = parent, origin_transform(joint)
    roots = links - parents.keys()
    if len(roots) != 1:
        raise ValueError("Expected one URDF root link")
    transforms = {next(iter(roots)): np.eye(4)}
    pending = dict(parents)
    while pending:
        ready = [child for child, (parent, _) in pending.items() if parent in transforms]
        if not ready:
            raise ValueError("Cycle or disconnected component in URDF")
        for child in ready:
            parent, local = pending.pop(child)
            transforms[child] = transforms[parent] @ local
    return transforms


def numbers(values):
    return " ".join(f"{float(value):.17g}" for value in values)


def convert_object(object_id, source_root, output_root):
    if object_id not in OBJECT_IDS:
        raise ValueError(f"Invalid GraspNet object ID: {object_id}")
    source = (source_root / object_id).resolve()
    urdf_source = source / "coacd/coacd.urdf"
    robot = ET.parse(urdf_source).getroot()
    transforms = link_transforms(robot)
    collisions = []
    for link in robot.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected convex collision meshes: {object_id}")
            path = (urdf_source.parent / mesh.get("filename")).resolve()
            if not path.is_relative_to(urdf_source.parent) or not path.is_file():
                raise ValueError(f"Invalid or missing convex mesh: {path}")
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"Invalid mesh scale: {path}")
            collisions.append((path, scale, transforms[link.get("name")] @ origin_transform(collision)))
    if not collisions:
        raise ValueError(f"Object has no convex collision geometry: {object_id}")
    visual_source = source / "simplified.obj"
    if not visual_source.is_file():
        raise ValueError(f"Official simplified surface mesh is missing: {object_id}")
    mesh = trimesh.load(visual_source, force="mesh", process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid surface mesh: {object_id}")

    destination = output_root / "processed_data" / object_id
    mesh_directory, urdf_directory = destination / "mesh", destination / "urdf"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(visual_source, mesh_directory / "simplified.obj")
    shutil.copytree(urdf_source.parent, urdf_directory, dirs_exist_ok=True)
    xml = ET.Element("mujoco", model=f"graspnet_{object_id}")
    ET.SubElement(xml, "compiler", angle="radian", meshdir=".")
    asset = ET.SubElement(xml, "asset")
    ET.SubElement(asset, "mesh", name="surface", file="../mesh/simplified.obj")
    body = ET.SubElement(ET.SubElement(xml, "worldbody"), "body", name="object")
    ET.SubElement(body, "geom", name="object_visual_0", type="mesh", mesh="surface",
                  contype="0", conaffinity="0", density="0")
    for index, (path, scale, transform) in enumerate(collisions):
        name = f"convex_{index:03d}"
        ET.SubElement(asset, "mesh", name=name, file=str(path.relative_to(urdf_source.parent)),
                      scale=numbers(scale))
        xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        ET.SubElement(body, "geom", name=f"object_collision_{index}", type="mesh", mesh=name,
                      pos=numbers(transform[:3, 3]), quat=numbers(xyzw[[3, 0, 1, 2]]), density="1")
    ET.indent(xml)
    xml_path = urdf_directory / "coacd.xml"
    collision_only_meshes = []
    while True:
        ET.ElementTree(xml).write(xml_path, encoding="unicode")
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            break
        except ValueError as error:
            tiny = re.search(r"mesh volume is too small: (convex_\d+)\s", str(error))
            if tiny is None or tiny[1] in collision_only_meshes:
                raise
            name = tiny[1]
            # Keep sub-resolution convex pieces for collisions. Shell inertia avoids
            # MuJoCo's volume threshold; explicit zero mass prevents assigning surface
            # density when the native environment overrides volumetric density.
            asset.find(f"mesh[@name='{name}']").set("inertia", "shell")
            body.find(f"geom[@mesh='{name}']").set("mass", "0")
            collision_only_meshes.append(name)
    body_id = model.body("object").id
    mass = float(model.body_mass[body_id])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid convex geometry mass: {object_id}")
    info = {"gravity_center": model.body_ipos[body_id].tolist(),
            "obb": mesh.bounding_box_oriented.extents.tolist(), "scale": 1.0,
            "density": 1.0, "mass": mass}
    info_path = destination / "info/simplified.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    scene_path = output_root / "scene_cfg" / object_id / "floating/scale001.npy"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    relative = lambda p: os.path.relpath(p, scene_path.parent)
    scene = {"scene": {object_id: {"type": "rigid_object",
        "file_path": relative(mesh_directory / "simplified.obj"), "xml_path": relative(xml_path),
        "urdf_path": relative(urdf_directory / "coacd.urdf"), "info_path": relative(info_path),
        "scale": np.ones(3), "pose": np.asarray([0., 0., 0., 1., 0., 0., 0.])}},
        "scene_id": f"{object_id}/floating/scale001",
        "task": {"type": "force_closure", "obj_name": object_id}}
    with scene_path.with_suffix(".npy.tmp").open("wb") as stream:
        np.save(stream, scene)
    scene_path.with_suffix(".npy.tmp").replace(scene_path)
    return {"object_id": object_id, "collision_mesh_count": len(collisions),
            "collision_only_meshes": collision_only_meshes,
            "surface_extent_m": np.ptp(mesh.vertices, axis=0).tolist(),
            "scene_scale": [1., 1., 1.], "scene_path": str(scene_path.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO_ROOT / "downloads/graspnet_1billion_meshdata.tar.gz")
    parser.add_argument("--raw-output", type=Path, default=REPO_ROOT / "assets/object/graspnet_1billion/raw_source")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets/object/graspnet_1billion")
    parser.add_argument("--download", action="store_true", help="Download or resume the official model archive")
    args = parser.parse_args()
    if args.download:
        download_archive(args.archive.resolve())
    digest = archive_digest(args.archive.resolve())
    source = extract_geometry(args.archive.resolve(), args.raw_output.resolve(), digest)
    output = args.output.resolve()
    records = [convert_object(object_id, source, output) for object_id in OBJECT_IDS]
    (output / "valid_split").mkdir(parents=True, exist_ok=True)
    (output / "valid_split/all.json").write_text(json.dumps(list(OBJECT_IDS), indent=2) + "\n")
    manifest = {"source_url": ARCHIVE_URL, "archive_sha256": digest,
                "object_count": len(records), "object_ids": list(OBJECT_IDS),
                "scale_policy": "original mesh coordinates in meters; no normalization; one scale [1,1,1]",
                "collision_policy": "reuse official CoACD meshes and fixed URDF transforms",
                "tiny_mesh_policy": "retain collision geometry; shell inertia and zero mass only for pieces below MuJoCo's volume threshold",
                "objects": records}
    (output / "import_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared all {len(records)} GraspNet objects for native Dexonomy: {output}", flush=True)


if __name__ == "__main__":
    main()
