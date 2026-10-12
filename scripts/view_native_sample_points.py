#!/usr/bin/env python3
"""Plot and interactively inspect native surface sampling checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import trimesh
from transforms3d.quaternions import quat2mat

from dexonomy.util.file_util import load_scene_cfg


def load_sampling(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name].copy() for name in
                  ("points_world", "inward_normals_world", "frames_world", "roll_angles_rad")}
    count = int(metadata["point_count"])
    if arrays["points_world"].shape != (count, 3):
        raise ValueError("Invalid surface point dimensions")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Nonfinite sampling data")
    return metadata, arrays, load_scene_cfg(metadata["scene_path"])


def world_mesh(cfg):
    mesh = trimesh.load(cfg["file_path"], force="mesh")
    mesh.vertices = (np.asarray(mesh.vertices) * cfg["scale"]) @ quat2mat(cfg["pose"][3:]).T + cfg["pose"][:3]
    return mesh


def write_plot(path, metadata, arrays, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points = arrays["points_world"] * 100
    target = metadata["target"]
    meshes = {name: world_mesh(cfg) for name, cfg in scene["scene"].items()
              if cfg["type"] == "rigid_object"}
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    panels = [fig.add_subplot(2, 2, i, projection="3d") for i in (1, 2, 3)]
    scene_name = Path(metadata["scene_path"]).parent.name
    titles = [f"A  {scene_name}: sampled object in blue", f"B  All {len(points)} points: front view",
              f"C  All {len(points)} points: back / underside view"]
    angles = [(50, 135), (22, 135), (-20, -45)]
    selected = int(np.argmax(points[:, 2]))

    for index, (ax, title, camera) in enumerate(zip(panels, titles, angles)):
        ax.computed_zorder = False
        visible = meshes if index == 0 else {target: meshes[target]}
        vertices = []
        for name, mesh in visible.items():
            verts = mesh.vertices * 100
            vertices.append(verts)
            color = "#2874bd" if name == target else "#c9ced6"
            collection = Poly3DCollection(verts[mesh.faces], facecolors=color, edgecolors="none",
                                          alpha=0.32 if name == target else 0.45, zorder=1)
            ax.add_collection3d(collection)
        ax.scatter(*points.T, c="#ffbd24", s=19 if index else 12, edgecolors="#6e4500",
                   linewidths=0.3, depthshade=False, zorder=3)
        ax.scatter(*points[selected], c="#ef4444", s=65, edgecolors="white", linewidths=1.0,
                   depthshade=False, zorder=4)
        if index:
            for point_index in range(0, len(points), 16):
                ax.text(*points[point_index], str(point_index), fontsize=7, color="#3e2800")
        all_vertices = np.concatenate(vertices)
        center = (all_vertices.min(0) + all_vertices.max(0)) / 2
        radius = np.ptp(all_vertices, axis=0).max() * 0.57
        for setter, coordinate in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(coordinate - radius, coordinate + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*camera)
        ax.set_title(title, fontsize=13, loc="left", pad=8)
        ax.set_xlabel("X (cm)", fontsize=9)
        ax.set_ylabel("Y (cm)", fontsize=9)
        ax.set_zlabel("Z (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

    roll = fig.add_subplot(2, 2, 4, projection="polar")
    roll_angles = arrays["roll_angles_rad"]
    duplicate = np.allclose(arrays["frames_world"][:len(points)], arrays["frames_world"][-len(points):])
    for index, angle in enumerate(roll_angles):
        roll.annotate("", xy=(angle, 0.88), xytext=(angle, 0),
                      arrowprops=dict(arrowstyle="->", color="#2874bd", lw=2))
        if duplicate and index == len(roll_angles) - 1:
            continue
        label = f"0 / {len(roll_angles) - 1}" if index == 0 and duplicate else str(index)
        roll.text(angle, 1.04, label, ha="center", va="center", fontsize=12,
                  color="#c2410c" if index == 0 else "#1e3a5f")
    roll.set_ylim(0, 1.18)
    roll.set_yticks([])
    roll.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    roll.set_xticklabels([])
    roll.grid(alpha=0.2)
    roll.set_title(f"D  Roll around surface normal at point {selected}", fontsize=13, pad=22)
    detail = "Native angle entries: 8\nEntries 0 and 7 overlap (−180° and +180°)" if duplicate and len(roll_angles) == 8 else f"Native angle entries: {len(roll_angles)}"
    fig.text(0.73, 0.04, detail, ha="center", fontsize=11, color="#c2410c")
    fig.suptitle(f"Native surface sampling — {len(points)} points × {len(roll_angles)} roll entries", fontsize=20, y=0.985)
    subtitle = "Fresh native sampling demonstration; historical raw seeds were not archived." if metadata["source"] == "fresh_native_sample" else "Original surface seeds saved before template matching."
    fig.text(0.5, 0.951, f"Target: {target}  |  {subtitle}", ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.89, bottom=0.08, wspace=0.10, hspace=0.27)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved point visualization: {path}", flush=True)


def show_viewer(metadata, arrays, scene):
    import glfw
    import mujoco
    import mujoco.viewer
    from view_clutter_grasps_mujoco import AllGraspsModel, SceneData

    target = metadata["target"]
    scene_path = Path(metadata["scene_path"])
    description = SceneData(scene_path.parent.name, [], scene_path, scene, [])
    view = AllGraspsModel(description, "assets/hand/shadow/right.xml", 0.0)
    for geom_id in range(view.model.ngeom):
        geom = view.model.geom(geom_id)
        if int(geom.group[0]) == 5:
            continue
        if geom.name.startswith("scene-" + target + "-"):
            view.model.geom_matid[geom_id] = -1
            view.model.geom_rgba[geom_id] = [0.12, 0.42, 0.8, 0.78]
        elif geom.name.startswith("scene-") and geom.name != "scene-table":
            view.model.geom_matid[geom_id] = -1
            view.model.geom_rgba[geom_id] = [0.65, 0.68, 0.72, 0.22]
    mujoco.mj_forward(view.model, view.data)
    points = arrays["points_world"]
    selected = int(np.argmax(points[:, 2]))
    state = {"normals": False, "labels": False, "rolls": False, "quit": False}

    def on_key(key):
        mapping = {glfw.KEY_N: "normals", glfw.KEY_L: "labels", glfw.KEY_R: "rolls"}
        if key in mapping:
            state[mapping[key]] = not state[mapping[key]]
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            state["quit"] = True

    def sphere(scn, position, color, label="", radius=0.0008):
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), position,
                          np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        geom.label = label
        scn.ngeom += 1

    def arrow(scn, start, end, color):
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                          np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.00035, start, end)
        scn.ngeom += 1

    print(f"Viewer: {len(points)} original surface points. Red point = {selected}.", flush=True)
    print("Mouse: rotate/zoom | N: outward normals | L: point indices | R: roll axes | Q: close", flush=True)
    with mujoco.viewer.launch_passive(view.model, view.data, key_callback=on_key,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        print("Point viewer window is open.", flush=True)
        viewer.cam.lookat[:] = points.mean(0)
        viewer.cam.distance = max(float(np.ptp(points, axis=0).max()) * 3.3, 0.20)
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running() and not state["quit"]:
            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                for index, point in enumerate(points):
                    color = [0.95, 0.15, 0.1, 1] if index == selected else [1, 0.76, 0.06, 1]
                    sphere(scn, point, color, str(index) if state["labels"] else "")
                    if state["normals"]:
                        arrow(scn, point, point - arrays["inward_normals_world"][index] * 0.008,
                              [0.18, 0.7, 0.4, 1])
                if state["rolls"]:
                    sphere(scn, points[selected], [1, 0.1, 0.1, 1], radius=0.0014)
                    for roll_index in range(len(arrays["roll_angles_rad"])):
                        direction = arrays["frames_world"][roll_index * len(points) + selected, :, 1]
                        arrow(scn, points[selected], points[selected] + direction * 0.008,
                              [0.1, 0.55, 0.95, 1])
            viewer.sync()
            time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", required=True, help="Raw native sampling .npz checkpoint.")
    parser.add_argument("--png", help="Save a multi-view PNG; defaults alongside the checkpoint.")
    parser.add_argument("--view", action="store_true", help="Open an interactive MuJoCo window.")
    args = parser.parse_args()
    source = Path(args.sampling).resolve()
    metadata, arrays, scene = load_sampling(source)
    write_plot(Path(args.png).resolve() if args.png else source.with_suffix(".png"), metadata, arrays, scene)
    if args.view:
        show_viewer(metadata, arrays, scene)


if __name__ == "__main__":
    main()
