#!/usr/bin/env python3
"""Visualize palm-position FPS grasps for one object at one physical scale."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.distance import pdist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.analyze_grasp_diversity import load_group
from scripts.view_clutter_grasps_mujoco import (
    AllGraspsModel, GraspRecord, PHASES, SceneData, _farthest_point_downsample,
)

DEFAULT_OBJECT = "core_bottle_1071fa4cddb2da2fc8724d5673a063a6"


def load_object_scale(library, object_id, scale):
    matches = []
    for path in sorted((library / "scene_cfg" / object_id / "floating").glob("*.npy")):
        scene = np.load(path, allow_pickle=True).item()
        target = scene["task"]["obj_name"]
        if set(scene["scene"]) != {target} or target != object_id:
            raise ValueError(f"Expected an isolated object: {path}")
        if np.allclose(scene["scene"][target]["scale"], scale, rtol=1e-6, atol=0):
            matches.append((path, scene))
    if len(matches) != 1:
        raise ValueError(f"Expected one object config at scale {scale}, found {len(matches)}")
    scene_path, scene = matches[0]
    relative = str(scene_path.relative_to(library / "scene_cfg").with_suffix(""))
    runtime = json.loads((library / "runtime.json").read_text())
    taxonomies = json.loads(Path(runtime["run_config"]).read_text())["taxonomies"]
    group = load_group(library, relative, taxonomies, "grasp")
    records = []
    for taxonomy in taxonomies:
        path = library / "succ_collect" / taxonomy / f"{relative}.npy"
        saved = np.load(path, allow_pickle=True).item()
        arrays = {phase: np.asarray(saved[f"{phase}_qpos"], dtype=float) for phase in PHASES}
        count = len(arrays["grasp"])
        if any(value.shape != (count, 29) for value in arrays.values()):
            raise ValueError(f"Inconsistent saved phases: {path}")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"Invalid saved grasp: {path}")
        records.extend(GraspRecord(taxonomy, object_id, path, index,
                                   {phase: value[index].copy() for phase, value in arrays.items()})
                       for index in range(count))
    if not records:
        raise ValueError("This object and scale have no successful grasps")
    if not np.array_equal(group["qpos"], np.asarray([record.qposes["grasp"] for record in records])):
        raise ValueError("The source library changed during loading; retry to capture one snapshot")
    return scene_path, scene, relative, group, records


def prepare_render(view, colors):
    model, data = view.model, view.data
    owners = np.full(model.ngeom, -2, dtype=int)  # -2 hidden; -1 object; >=0 hand
    world_vertices = {}
    for geom in range(model.ngeom):
        if model.geom_rgba[geom, 3] == 0:
            model.geom_group[geom] = 5
            continue
        # Attached bodies have names even when their visual geoms are unnamed.
        name = model.body(int(model.geom_bodyid[geom])).name
        owner = int(name.split("-")[1]) if name.startswith("grasp-") else -1
        owners[geom] = owner
        model.geom_group[geom] = 0 if owner == -1 else 1
        model.geom_matid[geom] = -1
        model.geom_rgba[geom] = [*(colors[owner] if owner >= 0 else [.20, .72, .46]), 1.]
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Camera bounds require mesh visual geometry")
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        world_vertices.setdefault(owner, []).append(
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return owners, {owner: np.concatenate(parts) for owner, parts in world_vertices.items()}


def camera_bounds(parts):
    vertices = np.concatenate(parts)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return (low+high)/2, float(np.linalg.norm(high-low))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO_ROOT / "output/single_object_grasps_dgn_shadow")
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--scale", type=float, default=.05)
    parser.add_argument("--max-grasps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--phase", choices=PHASES, default="grasp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hand-xml", type=Path, default=REPO_ROOT / "assets/hand/shadow/right.xml")
    args = parser.parse_args()
    if args.max_grasps < 1 or not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("max-grasps and scale must be positive")
    library = args.library.resolve()
    scene_path, config, relative, group, all_records = load_object_scale(library, args.object, args.scale)
    selected = _farthest_point_downsample(all_records, args.max_grasps, args.seed)
    identities = {id(record): index for index, record in enumerate(all_records)}
    indices = np.asarray([identities[id(record)] for record in selected], dtype=np.int32)
    positions = np.asarray([record.qposes["grasp"][:3] for record in all_records])
    sampled_positions = positions[indices]
    minimum_spacing = float(pdist(sampled_positions).min()*1000) if len(selected) > 1 else None
    spacing_text = f"{minimum_spacing:.1f} mm" if minimum_spacing is not None else "not applicable for one grasp"
    coverage_radius = float(np.linalg.norm(positions[:, None]-sampled_positions[None], axis=-1).min(axis=1).max()*1000)
    palette = plt.get_cmap("tab20")(np.asarray([0, 2, 6, 8, 10, 12, 14, 16, 18, 1, 3, 7]))[:, :3]
    colors = palette[np.arange(len(selected)) % len(palette)]
    scene = SceneData(relative, [], scene_path, config, selected)
    view = AllGraspsModel(scene, str(args.hand_xml), hand_alpha=1.)
    view.set_phase(args.phase)
    owners, vertices = prepare_render(view, colors)
    object_vertices = vertices[-1]
    individual_bounds = [camera_bounds([object_vertices, vertices[index]]) for index in range(len(selected))]
    overview_center, overview_span = camera_bounds(list(vertices.values()))
    individual_distance = max(span for _, span in individual_bounds)*1.25
    view.model.vis.global_.offwidth = 840
    view.model.vis.global_.offheight = 640
    view.model.vis.headlight.ambient[:] = [.35, .35, .35]
    view.model.vis.headlight.diffuse[:] = [.65, .65, .65]
    camera = mujoco.MjvCamera()
    camera.elevation = -20.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    output = args.output or REPO_ROOT / "output/grasp_fps_visualization" / args.object / f"scale_{args.scale:g}"
    output.mkdir(parents=True, exist_ok=True)

    def set_visible(hand):
        model = view.model
        mask = owners >= 0
        if hand is None:
            model.geom_group[mask] = 1
            model.geom_rgba[mask, 3] = .14
        else:
            model.geom_group[mask] = np.where(owners[mask] == hand, 1, 5)
            model.geom_rgba[mask, 3] = 1.

    # Individual poses use the same scale and two camera angles. No physics is stepped.
    columns, rows = 3, math.ceil(len(selected)/3)
    figure, axes = plt.subplots(rows, columns, figsize=(18, rows*2.9+1.4), squeeze=False,
                                facecolor="#f8fafc")
    with mujoco.Renderer(view.model, height=320, width=380) as renderer:
        for index, (record, ax) in enumerate(zip(selected, axes.flat)):
            set_visible(index)
            camera.lookat[:] = individual_bounds[index][0]
            camera.distance = individual_distance
            pair = []
            for azimuth in (125., 305.):
                camera.azimuth = azimuth
                renderer.update_scene(view.data, camera=camera, scene_option=option)
                pair.append(renderer.render().copy())
            ax.imshow(np.concatenate(pair, axis=1))
            ax.axis("off")
            ax.set_title(f"#{index+1:02d}  {record.taxonomy}\nSource row {record.source_index}",
                         fontsize=12, color=colors[index]*.75, pad=8)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis("off")
    figure.suptitle(f"Farthest-point sampled grasps | {args.object}\n"
                   f"Scale {args.scale:g} | {len(selected)} of {len(all_records)} saved grasps | "
                   f"FPS on grasp palm XYZ | seed {args.seed}", fontsize=16, y=.985)
    figure.text(.5, .022,
                f"Green: object. Each card shows two opposite views of one saved {args.phase} pose.\n"
                f"Minimum sampled palm spacing: {spacing_text}. "
                "Original grasp library unchanged.", ha="center", fontsize=12)
    figure.subplots_adjust(top=.89, bottom=.065, left=.02, right=.98, hspace=.28, wspace=.04)
    figure.savefig(output / "fps_grasps.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)

    set_visible(None)
    camera.lookat[:] = overview_center
    camera.distance = overview_span*1.15
    with mujoco.Renderer(view.model, height=640, width=840) as renderer:
        def render_overview(azimuth):
            camera.azimuth = azimuth
            renderer.update_scene(view.data, camera=camera, scene_option=option)
            return renderer.render().copy()

        figure, axes = plt.subplots(1, 2, figsize=(14, 6.3), facecolor="#f8fafc")
        for ax, angle in zip(axes, (125., 305.)):
            ax.imshow(render_overview(angle))
            ax.axis("off")
        figure.suptitle(f"One object, scale {args.scale:g} | {len(selected)} FPS grasps overlaid", fontsize=16)
        figure.text(.5, .045, "Green: object. Hands shown at 14% opacity in their saved positions.",
                    ha="center", fontsize=12)
        figure.subplots_adjust(top=.88, bottom=.10, left=.02, right=.98, wspace=.03)
        figure.savefig(output / "fps_scene.png", dpi=150, facecolor=figure.get_facecolor())
        plt.close(figure)
        frames = [render_overview(float(angle)) for angle in np.linspace(125, 485, 36, endpoint=False)]
        imageio.mimsave(output / "fps_scene_turntable.gif", frames, duration=120, loop=0)

    origins = [{"display_index": index+1, "taxonomy": record.taxonomy,
                "published_file": str(record.source_path), "published_row_zero_based": record.source_index,
                "aggregate_index_zero_based": int(indices[index]),
                "grasp_palm_xyz_m": record.qposes["grasp"][:3].tolist()}
               for index, record in enumerate(selected)]
    result = {"object_id": args.object, "scale": group["scale"], "relative": relative,
              "scene_path": str(scene_path), "input_grasps": len(all_records),
              "selected_count": len(selected), "seed": args.seed, "render_phase": args.phase,
              "fps_distance": "Euclidean distance between grasp palm XYZ positions",
              "fps_rule": "argmax over candidates of minimum distance to previously selected grasps",
              "minimum_sampled_palm_spacing_mm": minimum_spacing,
              "maximum_input_to_sampled_palm_distance_mm": coverage_radius,
              "source_grasp_qpos_sha256": group["qpos_sha256"], "selected_grasps": origins}
    (output / "fps_selection.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez(output / "selected_grasps.npz",
             **{f"{phase}_qpos": np.asarray([record.qposes[phase] for record in selected]) for phase in PHASES},
             taxonomy=np.asarray([record.taxonomy for record in selected]),
             published_row=np.asarray([record.source_index for record in selected]),
             aggregate_indices=indices, scale=group["scale"])
    print(json.dumps({"output": str(output), "input_grasps": len(all_records),
                      "selected_count": len(selected), "minimum_palm_spacing_mm": minimum_spacing,
                      "selected_taxonomies": sorted({record.taxonomy for record in selected})}, indent=2))


if __name__ == "__main__":
    main()
