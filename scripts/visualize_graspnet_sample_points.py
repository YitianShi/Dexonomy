#!/usr/bin/env python3
"""Visualize 128 native surface samples on 15 distinct GraspNet objects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
from transforms3d.quaternions import quat2mat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dexonomy.data.obj_loader import sample_init_pose
from dexonomy.util.file_util import load_scene_cfg


def draw_object(ax, record, camera=(24, 135), points_only=False, title=""):
    mesh = record["mesh"]
    vertices, points = mesh.vertices * 100, record["points_world"] * 100
    ax.computed_zorder = False
    if not points_only:
        ax.add_collection3d(Poly3DCollection(
            vertices[mesh.faces], facecolors="#4283b5", edgecolors="none",
            alpha=0.18, zorder=1))
    ax.scatter(*points.T, s=19, c="#f5a623", edgecolors="#75470b",
               linewidths=0.35, depthshade=False, zorder=3)
    center = (vertices.min(0) + vertices.max(0)) / 2
    radius = float(np.ptp(vertices, axis=0).max()) * 0.56
    for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(coordinate - radius, coordinate + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(*camera)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, color="#18334a", loc="left", pad=-2)


def write_plots(output, records, point_count, seed):
    columns = min(5, len(records))
    rows = math.ceil(len(records) / columns)
    fig = plt.figure(figsize=(columns * 4, rows * 3.6 + 1.1), facecolor="white")
    for number, record in enumerate(records, 1):
        ax = fig.add_subplot(rows, columns, number, projection="3d")
        draw_object(ax, record, title=f"{number:02d}  Object {record['id']}")
        span = np.ptp(record["mesh"].vertices, axis=0).max() * 100
        ax.text2D(0.02, 0.015, f"{point_count} points  |  span {span:.1f} cm",
                  transform=ax.transAxes, color="#526374", fontsize=9)
    fig.suptitle(f"GraspNet surface sampling  |  {len(records)} objects × {point_count} points",
                 fontsize=23, color="#18334a", y=0.976)
    fig.text(0.5, 0.926, f"Random selection without replacement · seed {seed} · original physical size · native Dexonomy sampler",
             ha="center", fontsize=11, color="#526374")
    fig.text(0.5, 0.018, "Gold: sampled surface points   |   Blue: translucent simplified mesh   |   Each object fitted to its panel",
             ha="center", fontsize=11, color="#526374")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.875, bottom=0.045,
                        hspace=0.10, wspace=0.02)
    fig.savefig(output / "overview.png", dpi=160, facecolor="white")
    plt.close(fig)
    individual = output / "individual"
    individual.mkdir(exist_ok=True)
    for number, record in enumerate(records, 1):
        fig = plt.figure(figsize=(14, 5), facecolor="white")
        for panel, (camera, points_only, title) in enumerate([
            ((24, 135), False, "Front / top"),
            ((-22, -45), False, "Back / underside"),
            ((24, 135), True, "Point cloud only"),
        ], 1):
            draw_object(fig.add_subplot(1, 3, panel, projection="3d"), record,
                        camera, points_only, title)
        fig.suptitle(f"GraspNet object {record['id']}  |  {point_count} surface points",
                     fontsize=16, color="#18334a")
        fig.text(0.5, 0.025, "Same points in every view · original physical size · points drawn through translucent mesh",
                 ha="center", fontsize=10, color="#526374")
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.075, top=0.82, wspace=0.02)
        fig.savefig(individual / f"{number:02d}_{record['id']}.png", dpi=150)
        plt.close(fig)


def write_html(output, records, metadata):
    objects = []
    for record in records:
        mesh = record["mesh"]
        objects.append({
            "id": record["id"], "seed": record["metadata"]["seed"],
            "scale_name": "scale001 (original size)",
            "center": mesh.bounds.mean(0).tolist(),
            "span": float(np.ptp(mesh.vertices, axis=0).max()),
            "vertices": np.asarray(mesh.vertices).round(8).tolist(),
            "faces": np.asarray(mesh.faces).tolist(),
            "points": record["points_world"].round(8).tolist(),
            "normals": (-record["inward_normals_world"]).round(8).tolist(),
        })
    template = Path(__file__).with_name("surface_sampling_viewer.html").read_text(encoding="utf-8")
    payload = json.dumps({**metadata, "objects": objects}, separators=(",", ":"))
    (output / "interactive.html").write_text(
        template.replace("__PAYLOAD__", payload.replace("<", "\\u003c")), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, default=REPO_ROOT / "assets/object/graspnet_1billion/scene_cfg")
    parser.add_argument("--objects", type=int, default=15)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output/graspnet_sampling_15x128")
    args = parser.parse_args()
    if min(args.objects, args.points) < 1 or args.seed < 0:
        parser.error("objects and points must be positive; seed must be nonnegative")
    root, output = args.object_root.resolve(), args.output.resolve()
    if root == output or root in output.parents or output in root.parents:
        parser.error("output must be separate from the input assets")
    catalog = sorted(root.glob("*/floating/scale001.npy"))
    if len(catalog) < args.objects:
        parser.error(f"Found {len(catalog)} distinct objects, need {args.objects}: {root}")
    chosen = np.random.default_rng(args.seed).choice(len(catalog), args.objects, replace=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sampling").mkdir(exist_ok=True)
    records = []
    for number, index in enumerate(chosen, 1):
        scene_path = catalog[int(index)]
        scene = load_scene_cfg(str(scene_path))
        name = scene["task"]["obj_name"]
        if name != scene_path.parents[1].name or name not in {f"{i:03d}" for i in range(88)}:
            raise ValueError(f"Expected a GraspNet object ID 000–087: {scene_path}")
        body = scene["scene"][name]
        scale, pose = np.asarray(body["scale"], dtype=float), np.asarray(body["pose"], dtype=float)
        if (body["type"] != "rigid_object" or scale.shape != (3,) or
                not np.allclose(scale, np.ones(3)) or pose.shape != (7,) or
                not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-12):
            raise ValueError(f"Expected a rigid GraspNet mesh with original meter coordinates: {scene_path}")
        mesh = trimesh.load(body["file_path"], force="mesh")
        sample_seed = int(np.random.SeedSequence([args.seed, number]).generate_state(1)[0])
        np.random.seed(sample_seed)
        frames_local, points_local = sample_init_pose(mesh, args.points, 1)
        rotation = quat2mat(pose[3:])
        points_world = (points_local * scale) @ rotation.T + pose[:3]
        frames_world = rotation[None] @ frames_local
        inward_normals = frames_world[:, :, 0]
        if (points_world.shape != (args.points, 3) or
                not all(np.isfinite(a).all() for a in (points_world, frames_world)) or
                len(np.unique(points_local, axis=0)) != args.points):
            raise ValueError(f"Invalid native surface samples: {name}")
        # Trimesh's triangle tolerances can misclassify tiny meter-scale faces.
        # Normalize only the distance check; sampling and exported geometry stay
        # in the original physical coordinates.
        center = mesh.bounds.mean(0)
        span = float(np.ptp(mesh.vertices, axis=0).max())
        if not np.isfinite(span) or span <= 0:
            raise ValueError(f"Invalid surface bounds: {name}")
        check_mesh = mesh.copy()
        check_mesh.vertices = (mesh.vertices - center) / span
        _, distance, _ = trimesh.proximity.closest_point(check_mesh, (points_local - center) / span)
        distance *= span
        if distance.max() > span * 1e-7:
            raise ValueError(f"Samples are not on the source surface: {name}")
        metadata = {
            "dataset": "GraspNet-1Billion", "target": name, "scene_path": str(scene_path),
            "mesh_path": str(Path(body["file_path"]).resolve()),
            "source": "fresh_native_sample", "sampler": "dexonomy.data.obj_loader.sample_init_pose",
            "point_count": args.points, "seed": sample_seed, "selection_seed": args.seed,
            "scale": scale.tolist(), "pose": pose.tolist(), "scale_name": "scale001",
            "units": "meters", "max_surface_error_m": float(distance.max()),
            "trimesh_version": trimesh.__version__, "numpy_version": np.__version__,
        }
        np.savez_compressed(output / "sampling" / f"{name}.npz",
                            points_local=points_local, points_world=points_world,
                            inward_normals_world=inward_normals, frames_world=frames_world,
                            roll_angles_rad=np.array([-np.pi]), metadata=json.dumps(metadata))
        world_mesh = mesh.copy()
        world_mesh.vertices = (mesh.vertices * scale) @ rotation.T + pose[:3]
        records.append({"id": name, "mesh": world_mesh, "points_local": points_local,
                        "points_world": points_world, "inward_normals_world": inward_normals,
                        "metadata": metadata})
        print(f"[{number:02d}/{args.objects}] GraspNet {name}: {args.points} points, surface error {distance.max():.2e} m", flush=True)

    metadata = {"dataset": "GraspNet-1Billion", "seed": args.seed, "catalog_count": len(catalog),
                "object_count": len(records), "point_count": args.points,
                "total_points": len(records) * args.points, "scale_policy": "original meter coordinates, [1,1,1]",
                "sampler": "dexonomy.data.obj_loader.sample_init_pose", "units": "meters",
                "sampling_method": "sample_surface_even with sample_surface fallback to the requested count",
                "source": "fresh_native_sample"}
    np.savez_compressed(output / "sampling_points.npz", object_ids=np.array([r["id"] for r in records]),
                        points_local=np.stack([r["points_local"] for r in records]),
                        points_world=np.stack([r["points_world"] for r in records]),
                        inward_normals_world=np.stack([r["inward_normals_world"] for r in records]),
                        metadata=json.dumps(metadata))
    (output / "manifest.json").write_text(json.dumps(
        {**metadata, "objects": [r["metadata"] for r in records]}, indent=2), encoding="utf-8")
    write_plots(output, records, args.points, args.seed)
    write_html(output, records, metadata)
    print(f"Saved overview, individual views, interactive HTML and point archives: {output}", flush=True)


if __name__ == "__main__":
    main()
