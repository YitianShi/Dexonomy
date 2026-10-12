#!/usr/bin/env python3
"""Render the closest saved grasp pair for one audited object and physical scale."""

from __future__ import annotations

import argparse
import csv
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dexonomy.util.grasp_diversity import TOLERANCES, analyze_grasps, pose_pair_distances
from scripts.analyze_grasp_diversity import load_group
from scripts.view_clutter_grasps_mujoco import AllGraspsModel, GraspRecord, SceneData

DEFAULT_OBJECT = "core_bottle_1071fa4cddb2da2fc8724d5673a063a6"
COLORS = np.asarray([[.13, .43, .83], [.94, .43, .12]])


def source_record(library, relative, taxonomy, row):
    path = library / "succ_collect" / taxonomy / f"{relative}.npy"
    saved = np.load(path, allow_pickle=True).item()
    source = Path(str(saved["source_grasp_paths"][row]))
    candidate = int(saved["source_indices"][row])
    provenance = {"taxonomy": taxonomy, "published_file": str(path),
                  "published_row_zero_based": int(row), "attempt_archive": str(source),
                  "candidate_index_zero_based": candidate}
    with np.load(source, allow_pickle=False) as archive:
        matches = np.flatnonzero(archive["candidate_indices"] == candidate)
        if len(matches) != 1:
            raise ValueError(f"Candidate {candidate} has ambiguous provenance in {source}")
        index = int(matches[0])
        provenance.update(sample_point_index_zero_based=int(archive["sample_point_indices"][index]),
                          inplane_index_zero_based=int(archive["inplane_indices"][index]))
    return path, saved, provenance


def configure_geometry(view):
    """Assign distinct viewer groups; retain the original hand and object poses."""
    model, data = view.model, view.data
    vertices = []
    for index in range(model.ngeom):
        # Many hand geoms are unnamed; the attached body names retain the prefix.
        name = model.body(int(model.geom_bodyid[index])).name
        if model.geom_rgba[index, 3] == 0:
            model.geom_group[index] = 5
            continue
        if name.startswith("grasp-"):
            hand = int(name.split("-")[1])
            model.geom_group[index] = hand + 1
            color = COLORS[hand]
        else:
            model.geom_group[index] = 0
            color = [.21, .67, .46]
        model.geom_matid[index] = -1
        model.geom_rgba[index] = [*color, 1.]
        if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[index]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start+count]
            vertices.append(local @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index])
    bounds = np.asarray(np.concatenate(vertices))
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    return (low+high)/2, float(np.linalg.norm(high-low))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=REPO_ROOT / "output" / f"grasp_diversity_{DEFAULT_OBJECT}")
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--scale", type=float, default=.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hand-xml", type=Path, default=REPO_ROOT / "assets/hand/shadow/right.xml")
    args = parser.parse_args()
    summary = json.loads((args.report / "summary.json").read_text())
    with (args.report / "object_scale_metrics.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["object_id"] == args.object
                and np.allclose(json.loads(row["scale"]), args.scale, rtol=1e-6)]
    if len(rows) != 1:
        raise ValueError("Select exactly one audited object at one uniform physical scale")
    row = rows[0]
    library, run = Path(summary["source_library"]), Path(summary["source_checkpoint_run"])
    taxonomies = json.loads((run / "run_config.json").read_text())["taxonomies"]
    relative, phase = row["relative"], summary["phase"]
    group = load_group(library, relative, taxonomies, phase)
    if group["qpos_sha256"] != row["qpos_sha256"]:
        raise ValueError("The source library changed since the audit; rerun the audit first")
    raw = group["qpos"]
    valid = np.isfinite(raw).all(axis=1) & (np.linalg.norm(raw[:, 3:7], axis=1) > 1e-12)
    valid_indices = np.flatnonzero(valid)
    normalized = np.asarray(raw[valid], dtype=float).copy()
    normalized[:, 3:7] /= np.linalg.norm(normalized[:, 3:7], axis=1)[:, None]
    _, _, edges = analyze_grasps(raw[valid], group["labels"][valid], group["diagonal"], group["center"])
    if not len(edges):
        raise ValueError("No pair meets the default near-neighbor tolerances in this group")
    distances = np.asarray(pose_pair_distances(normalized, *edges.T, group["diagonal"]))
    tolerance = TOLERANCES["default"]
    limits = np.asarray([tolerance.position_fraction, tolerance.rotation_deg,
                         tolerance.joint_rms_deg, tolerance.joint_max_deg])
    scores = np.max(distances / limits[:, None], axis=0)
    # All omitted pairs have a component above its tolerance and hence score > 1.
    order = np.lexsort((edges[:, 1], edges[:, 0], scores))
    nearest = int(order[0])
    pair = valid_indices[edges[nearest]]
    position, rotation, rms, maximum = distances[:, nearest]
    metrics = {"position_mm": float(position*group["diagonal"]*1000),
               "rotation_deg": float(rotation), "joint_rms_deg": float(rms),
               "joint_max_deg": float(maximum), "score": float(scores[nearest]),
               "raw_qpos_identical": bool(np.array_equal(raw[pair[0]], raw[pair[1]]))}
    records, provenance = [], []
    for index in pair:
        taxonomy_index, source_row = map(int, group["locations"][index])
        taxonomy = taxonomies[taxonomy_index]
        path, saved, origin = source_record(library, relative, taxonomy, source_row)
        records.append(GraspRecord(taxonomy, group["object"], path, source_row,
                                   {phase: saved[f"{phase}_qpos"][source_row]}))
        provenance.append(origin)
    scene_path = library / "scene_cfg" / f"{relative}.npy"
    scene = SceneData(relative, [], scene_path,
                      np.load(scene_path, allow_pickle=True).item(), records)
    view = AllGraspsModel(scene, str(args.hand_xml), hand_alpha=1.)
    view.set_phase(phase)
    center, span = configure_geometry(view)
    view.model.vis.global_.offwidth = 540
    view.model.vis.global_.offheight = 480
    view.model.vis.headlight.ambient[:] = [.35, .35, .35]
    view.model.vis.headlight.diffuse[:] = [.65, .65, .65]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = center
    camera.distance = span * 1.1
    camera.elevation = -20.
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    output = args.output or args.report / f"nearest_pair_scale_{args.scale:g}"
    output.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(view.model, height=480, width=540) as renderer:
        def render_pair(azimuth):
            camera.azimuth = azimuth
            images = []
            for hand in range(2):
                option.geomgroup[1:3] = [hand == 0, hand == 1]
                renderer.update_scene(view.data, camera=camera, scene_option=option)
                images.append(renderer.render().copy())
            # Same-camera pixel superposition avoids coincident-mesh z-fighting.
            overlay = np.rint((images[0].astype(float)+images[1])/2).astype(np.uint8)
            return [*images, overlay]

        figure, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#f8fafc")
        for row_index, azimuth in enumerate((125., 305.)):
            images = render_pair(azimuth)
            for column, img in enumerate(images):
                ax = axes[row_index, column]
                ax.imshow(img)
                ax.axis("off")
                if not row_index:
                    title = (f"A: row {provenance[0]['published_row_zero_based']} (blue)",
                             f"B: row {provenance[1]['published_row_zero_based']} (orange)",
                             "A + B: exact camera overlay")[column]
                    ax.set_title(title, fontsize=14, pad=12)
                if not column:
                    ax.text(.02, .04, f"View {row_index+1}", transform=ax.transAxes,
                            color="white", fontsize=12)
        figure.suptitle(f"Closest grasp pair | {args.object}\n"
                        f"Object scale {args.scale:g} | {phase} | {records[0].taxonomy}",
                        fontsize=15, y=.98)
        figure.text(.5, .035,
                    f"Position: {metrics['position_mm']:.6g} mm    "
                    f"Rotation: {rotation:.6g} deg    Joint RMS: {rms:.6g} deg    "
                    f"Joint max: {maximum:.6g} deg\n"
                    "Green: object. Saved poses unchanged; no separation or difference magnification.",
                    ha="center", fontsize=12)
        figure.subplots_adjust(top=.87, bottom=.095, left=.02, right=.98, hspace=.025, wspace=.025)
        figure.savefig(output / "nearest_pair.png", dpi=130, facecolor=figure.get_facecolor())
        plt.close(figure)
        frames = [np.concatenate(render_pair(float(angle)), axis=1)
                  for angle in np.linspace(125, 485, 36, endpoint=False)]
        imageio.mimsave(output / "nearest_pair_turntable.gif", frames, duration=120, loop=0)

    result = {"object_id": args.object, "scale": group["scale"], "phase": phase,
              "relative": relative, "input_grasps": len(raw), "metrics": metrics,
              "minimum_score_tie_count": int(np.count_nonzero(scores == scores[nearest])),
              "ranking": "max(position_fraction/0.02, rotation_deg/10, joint_rms_deg/5, joint_max_deg/15)",
              "source_qpos_sha256": group["qpos_sha256"], "grasps": provenance,
              "render": "MuJoCo saved qpos; same camera for A, B and 50/50 pixel overlay"}
    (output / "nearest_pair.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez(output / "nearest_pair_qpos.npz", qpos=raw[pair], scale=group["scale"])
    print(json.dumps({"output": str(output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
