#!/usr/bin/env python3
"""Visualize configured samples of filtered clutter grasps at the same time."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from ruamel.yaml import YAML

from dexonomy.util.file_util import load_scene_cfg


PHASES = ("pregrasp", "grasp", "squeeze")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("clutter_grasp_viewer.yaml")
YAML_LOADER = YAML(typ="safe")
PALETTE = np.asarray(
    [
        [0.91, 0.30, 0.24],
        [0.20, 0.60, 0.86],
        [0.18, 0.80, 0.44],
        [0.95, 0.77, 0.06],
        [0.61, 0.35, 0.71],
        [0.90, 0.49, 0.13],
        [0.10, 0.74, 0.61],
        [0.93, 0.44, 0.58],
    ],
    dtype=float,
)


@dataclass
class GraspRecord:
    taxonomy: str
    target_name: str
    source_path: Path
    source_index: int
    qposes: dict[str, np.ndarray]


@dataclass
class SceneGroup:
    name: str
    directories: list[Path]


@dataclass
class SceneData:
    name: str
    directories: list[Path]
    scene_path: Path
    scene_cfg: dict
    grasps: list[GraspRecord]


def _load_viewer_config(path: str) -> dict:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Viewer config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = YAML_LOADER.load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Viewer config must contain a YAML mapping: {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    preliminary_args, _ = config_parser.parse_known_args()
    config = _load_viewer_config(preliminary_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Show the complete clutter scene and a configured number of its "
            "filtered grasps simultaneously in MuJoCo Viewer."
        )
    )
    parser.add_argument(
        "--config",
        default=preliminary_args.config,
        help=(
            "Viewer-only YAML configuration file "
            f"(default: {DEFAULT_CONFIG_PATH})."
        ),
    )
    parser.add_argument(
        "--root",
        default="output/clutter_mujoco_shadow/clutter/filtered_grasps",
        help="Root directory containing taxonomy/scene_xxxxxx/*.npy.",
    )
    parser.add_argument(
        "--taxonomy",
        default="all",
        help="Taxonomy folder to load; use 'all' to combine every taxonomy.",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Only load one scene (for example scene_000012 or 12).",
    )
    parser.add_argument("--start", type=int, default=0, help="First scene index to show.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum scenes to show; 0 means all."
    )
    parser.add_argument(
        "--phase",
        "--qpos",
        dest="phase",
        choices=PHASES,
        default="grasp",
        help="Initial hand pose phase. Keys 1/2/3 switch phase in the viewer.",
    )
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=int(config.get("max_grasps_per_target", 5)),
        help=(
            "Maximum hand poses per target object at its scene scale after "
            "farthest-point sampling; "
            "0 means every filtered grasp (default: viewer YAML)."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(config.get("sample_seed", 12)),
        help="Seed used to select the first farthest-point sample (default: viewer YAML).",
    )
    parser.add_argument(
        "--hand-alpha",
        type=float,
        default=0.35,
        help="Opacity of hand meshes in [0, 1].",
    )
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-30.0)
    parser.add_argument("--hand-xml", default="assets/hand/shadow/right.xml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compile selected scenes and validate qposes without opening a window.",
    )
    return parser.parse_args()


def _scene_selector(value: str) -> str:
    if value.isdigit():
        return f"scene_{int(value):06d}"
    return value


def collect_scene_groups(args: argparse.Namespace) -> tuple[list[SceneGroup], int]:
    root = Path(args.root)
    if not root.is_dir():
        candidates = sorted(Path("output").glob("*/clutter/filtered_grasps"))
        hint = ""
        if candidates:
            hint = "\nAvailable filtered grasp roots:\n  " + "\n  ".join(
                str(path) for path in candidates
            )
        raise FileNotFoundError(
            f"Filtered grasp root does not exist: {root}{hint}"
        )

    if args.taxonomy.lower() == "all":
        taxonomy_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    else:
        taxonomy_dir = root / args.taxonomy
        if not taxonomy_dir.is_dir():
            raise FileNotFoundError(
                f"Taxonomy directory does not exist: {taxonomy_dir}"
            )
        taxonomy_dirs = [taxonomy_dir]

    grouped: dict[str, list[Path]] = {}
    for taxonomy_dir in taxonomy_dirs:
        for scene_dir in sorted(taxonomy_dir.glob("scene_*")):
            if scene_dir.is_dir() and any(scene_dir.glob("*.npy")):
                grouped.setdefault(scene_dir.name, []).append(scene_dir)
    scene_groups = [
        SceneGroup(name=name, directories=directories)
        for name, directories in sorted(grouped.items())
    ]
    if args.scene:
        requested = _scene_selector(args.scene)
        scene_groups = [group for group in scene_groups if group.name == requested]
        if not scene_groups:
            available = ", ".join(sorted(grouped)) or "<none>"
            raise FileNotFoundError(
                f"Scene {requested!r} does not exist below {root}. "
                f"Available scenes: {available}"
            )

    total = len(scene_groups)
    scene_groups = scene_groups[max(0, args.start) :]
    if args.limit > 0:
        scene_groups = scene_groups[: args.limit]
    return scene_groups, total


def _resolve_scene_path(value: str, sample_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        # Older outputs sometimes store a path relative to the dataset file.
        candidate = sample_path.parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Scene config referenced by {sample_path} not found: {value}")


def _farthest_point_downsample(
    records: list[GraspRecord], max_grasps: int, seed: int
) -> list[GraspRecord]:
    """Sample spatially separated grasps using their grasp palm positions."""
    if max_grasps <= 0 or len(records) <= max_grasps:
        return records

    points = np.asarray(
        [record.qposes["grasp"][:3] for record in records], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    selected_indices = [int(rng.integers(len(records)))]
    min_squared_distances = np.full(len(records), np.inf, dtype=np.float64)
    selected_mask = np.zeros(len(records), dtype=bool)

    while len(selected_indices) < max_grasps:
        latest = selected_indices[-1]
        selected_mask[latest] = True
        squared_distances = np.sum((points - points[latest]) ** 2, axis=1)
        min_squared_distances = np.minimum(
            min_squared_distances, squared_distances
        )
        min_squared_distances[selected_mask] = -1.0
        selected_indices.append(int(np.argmax(min_squared_distances)))

    return [records[index] for index in selected_indices]


def _downsample_per_target(
    records: list[GraspRecord], max_grasps: int, seed: int
) -> list[GraspRecord]:
    """Run FPS independently for every target object in the scene."""
    if max_grasps <= 0:
        return records

    grouped: dict[str, list[GraspRecord]] = {}
    for record in records:
        grouped.setdefault(record.target_name, []).append(record)

    sampled: list[GraspRecord] = []
    for target_index, target_name in enumerate(sorted(grouped)):
        sampled.extend(
            _farthest_point_downsample(
                grouped[target_name], max_grasps, seed + target_index
            )
        )
    return sampled


def load_scene_data(
    scene_group: SceneGroup, max_grasps: int, sample_seed: int = 12
) -> SceneData:
    records: list[GraspRecord] = []
    scene_path: Path | None = None

    for scene_dir in scene_group.directories:
        taxonomy = scene_dir.parent.name
        for sample_path in sorted(scene_dir.glob("*.npy")):
            sample = np.load(sample_path, allow_pickle=True).item()
            if "scene_path" not in sample:
                raise KeyError(f"{sample_path} does not contain scene_path")
            this_scene_path = _resolve_scene_path(
                str(sample["scene_path"]), sample_path
            )
            if scene_path is None:
                scene_path = this_scene_path

            arrays: dict[str, np.ndarray] = {}
            for phase in PHASES:
                key = f"{phase}_qpos"
                if key not in sample:
                    raise KeyError(f"{sample_path} does not contain {key}")
                value = np.asarray(sample[key], dtype=float)
                if value.ndim == 1:
                    value = value[None]
                arrays[phase] = value

            count = min(len(arrays[phase]) for phase in PHASES)
            target_name = str(sample.get("target_name", sample_path.stem))
            source_indices = np.asarray(
                sample.get("source_indices", np.arange(count)), dtype=int
            ).reshape(-1)
            for index in range(count):
                records.append(
                    GraspRecord(
                        taxonomy=taxonomy,
                        target_name=target_name,
                        source_path=sample_path,
                        source_index=(
                            int(source_indices[index])
                            if index < len(source_indices)
                            else index
                        ),
                        qposes={
                            phase: arrays[phase][index].copy() for phase in PHASES
                        },
                    )
                )

    if scene_path is None:
        raise ValueError(f"No grasp files found for {scene_group.name}")
    if not records:
        raise ValueError(f"No grasp qposes found for {scene_group.name}")
    try:
        scene_number = int(scene_group.name.rsplit("_", 1)[-1])
    except ValueError:
        scene_number = 0
    records = _downsample_per_target(
        records, max_grasps, sample_seed + scene_number
    )

    return SceneData(
        name=scene_group.name,
        directories=scene_group.directories,
        scene_path=scene_path,
        scene_cfg=load_scene_cfg(str(scene_path)),
        grasps=records,
    )


def _qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


class AllGraspsModel:
    """A static scene with one visual-only hand instance per grasp."""

    def __init__(self, scene: SceneData, hand_xml: str, hand_alpha: float):
        self.scene = scene
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.getcwd()
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.spec.worldbody.add_light(
            name="viewer-key-light", pos=[-0.5, -0.5, 1.5], castshadow=False
        )
        self.spec.worldbody.add_light(
            name="viewer-fill-light", pos=[0.5, 0.4, 1.0], castshadow=False
        )

        centers = []
        for object_name, object_cfg in scene.scene_cfg["scene"].items():
            object_type = object_cfg["type"]
            if object_type == "rigid_object":
                self._add_rigid_object(object_name, object_cfg)
                centers.append(np.asarray(object_cfg["pose"][:3], dtype=float))
            elif object_type == "plane":
                self._add_plane(object_name, object_cfg)
            else:
                raise NotImplementedError(
                    f"All-grasp viewer does not support scene object type {object_type!r}"
                )

        target_names = sorted({record.target_name for record in scene.grasps})
        target_colors = {
            name: PALETTE[index % len(PALETTE)]
            for index, name in enumerate(target_names)
        }
        self.target_colors = target_colors
        self._hand_layouts: list[list[tuple[str, int, int]]] = []
        for index, record in enumerate(scene.grasps):
            self._add_hand(
                hand_xml,
                index,
                target_colors[record.target_name],
                float(np.clip(hand_alpha, 0.0, 1.0)),
            )

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.center = (
            np.mean(np.stack(centers), axis=0)
            if centers
            else np.asarray([0.0, 0.0, 0.05])
        )
        self.center[2] = max(float(self.center[2]), 0.05)
        self.phase = ""

    @staticmethod
    def _absolute_mesh_paths(child_spec: mujoco.MjSpec, xml_path: str) -> None:
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        for mesh in child_spec.meshes:
            mesh.file = os.path.join(xml_dir, child_spec.meshdir, mesh.file)

    @staticmethod
    def _prepare_visual_geoms(
        child_spec: mujoco.MjSpec, *, delete_collisions: bool
    ) -> None:
        geoms = list(child_spec.geoms)
        visual_geoms = [
            geom
            for geom in geoms
            if int(geom.contype) == 0 and int(geom.conaffinity) == 0
        ]
        # Keep collision geometry as a fallback for assets with no visual mesh.
        if visual_geoms and delete_collisions:
            for geom in geoms:
                if geom not in visual_geoms:
                    geom.delete()
        for geom in child_spec.geoms:
            geom.contype = 0
            geom.conaffinity = 0
            if visual_geoms and geom not in visual_geoms:
                # Rigid-object collision meshes may provide their body mass, so
                # keep them in the model but put them in a hidden viewer group.
                geom.group = 5
                geom.rgba[3] = 0.0

    def _add_rigid_object(self, name: str, cfg: dict) -> None:
        xml_path = str(cfg["xml_path"])
        child_spec = mujoco.MjSpec.from_file(xml_path)
        self._absolute_mesh_paths(child_spec, xml_path)
        for mesh in child_spec.meshes:
            mesh.scale *= np.asarray(cfg["scale"], dtype=float)
        child_spec.meshdir = self.spec.meshdir
        self._prepare_visual_geoms(child_spec, delete_collisions=False)
        for actuator in list(child_spec.actuators):
            actuator.delete()

        frame = self.spec.worldbody.add_frame()
        child_world = frame.attach_body(child_spec.worldbody, f"scene-{name}-", "")
        child_world.pos = cfg["pose"][:3]
        child_world.quat = cfg["pose"][3:]

    def _add_plane(self, name: str, cfg: dict) -> None:
        self.spec.worldbody.add_geom(
            name=f"scene-{name}",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=cfg.get("pose", [0, 0, 0, 1, 0, 0, 0])[:3],
            quat=cfg.get("pose", [0, 0, 0, 1, 0, 0, 0])[3:],
            size=cfg.get("size", [0, 0, 1]),
            contype=0,
            conaffinity=0,
            rgba=[0.38, 0.34, 0.29, 1.0],
            group=0,
        )

    def _add_hand(
        self,
        hand_xml: str,
        index: int,
        color: np.ndarray,
        alpha: float,
    ) -> None:
        child_spec = mujoco.MjSpec.from_file(hand_xml)
        self._absolute_mesh_paths(child_spec, hand_xml)
        child_spec.meshdir = self.spec.meshdir
        joint_info = [(joint.name, _qpos_width(joint.type)) for joint in child_spec.joints]
        for actuator in list(child_spec.actuators):
            actuator.delete()
        self._prepare_visual_geoms(child_spec, delete_collisions=True)
        rgba = [float(color[0]), float(color[1]), float(color[2]), alpha]
        for material in child_spec.materials:
            material.rgba[:] = rgba
        for geom in child_spec.geoms:
            geom.rgba[:] = rgba
            geom.group = 1

        prefix = f"grasp-{index:05d}-"
        frame = self.spec.worldbody.add_frame()
        child_world = frame.attach_body(child_spec.worldbody, prefix, "")
        freejoint_name = f"{prefix}freejoint"
        child_world.add_freejoint(name=freejoint_name)

        layout = [(freejoint_name, 0, 7)]
        source_offset = 7
        for joint_name, width in joint_info:
            layout.append((prefix + joint_name, source_offset, width))
            source_offset += width
        self._hand_layouts.append(layout)

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase: {phase}")
        for record, layout in zip(self.scene.grasps, self._hand_layouts):
            source = record.qposes[phase]
            expected = max(offset + width for _, offset, width in layout)
            if len(source) != expected:
                raise ValueError(
                    f"{record.source_path} has qpos length {len(source)}, expected {expected}"
                )
            for joint_name, source_offset, width in layout:
                joint_id = self.model.joint(joint_name).id
                target_offset = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[target_offset : target_offset + width] = source[
                    source_offset : source_offset + width
                ]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.phase = phase


def _print_scene_summary(scene: SceneData, model: AllGraspsModel) -> None:
    counts: dict[str, int] = {}
    for record in scene.grasps:
        counts[record.target_name] = counts.get(record.target_name, 0) + 1
    taxonomy_count = len({record.taxonomy for record in scene.grasps})
    count_text = ", ".join(f"{name[:10]}…={count}" for name, count in counts.items())
    print(
        f"{scene.name}: {len(scene.grasps)} grasps from {len(counts)} targets "
        f"and {taxonomy_count} taxonomies; "
        f"model bodies={model.model.nbody}, geoms={model.model.ngeom}",
        flush=True,
    )
    print(f"  {count_text}", flush=True)


def show_scene(
    scene: SceneData,
    display_index: int,
    display_total: int,
    args: argparse.Namespace,
) -> tuple[int, bool]:
    import glfw
    import mujoco.viewer

    print(f"[{display_index + 1}/{display_total}] Building {scene.name} ...", flush=True)
    all_grasps = AllGraspsModel(scene, args.hand_xml, args.hand_alpha)
    all_grasps.set_phase(args.phase)
    _print_scene_summary(scene, all_grasps)

    state = {
        "scene_delta": 0,
        "phase": args.phase,
        "hands_visible": True,
        "toggle_hands": False,
        "quit": False,
    }

    def on_key(key: int) -> None:
        if key in (glfw.KEY_RIGHT, glfw.KEY_D):
            state["scene_delta"] = 1
        elif key in (glfw.KEY_LEFT, glfw.KEY_A):
            state["scene_delta"] = -1
        elif key == glfw.KEY_1:
            state["phase"] = "pregrasp"
        elif key == glfw.KEY_2:
            state["phase"] = "grasp"
        elif key == glfw.KEY_3:
            state["phase"] = "squeeze"
        elif key == glfw.KEY_H:
            state["toggle_hands"] = True
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            state["quit"] = True

    with mujoco.viewer.launch_passive(
        all_grasps.model,
        all_grasps.data,
        key_callback=on_key,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = all_grasps.center
        viewer.cam.distance = args.distance
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation

        while viewer.is_running():
            if state["quit"]:
                return 0, True
            if state["scene_delta"]:
                return int(state["scene_delta"]), False
            if state["phase"] != all_grasps.phase:
                all_grasps.set_phase(str(state["phase"]))
                print(f"  phase: {all_grasps.phase}", flush=True)
            if state["toggle_hands"]:
                state["toggle_hands"] = False
                state["hands_visible"] = not state["hands_visible"]
                viewer.opt.geomgroup[1] = state["hands_visible"]
            viewer.sync()
            time.sleep(0.01)
    return 1, False


def main() -> int:
    args = parse_args()
    if args.max_grasps < 0:
        print("--max-grasps must be non-negative", file=sys.stderr)
        return 2
    if not 0.0 <= args.hand_alpha <= 1.0:
        print("--hand-alpha must be between 0 and 1", file=sys.stderr)
        return 2

    try:
        scene_groups, total = collect_scene_groups(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not scene_groups:
        print("No matching clutter scenes found.", file=sys.stderr)
        return 1

    print(f"Found {total} matching scenes; selected {len(scene_groups)}.")
    if args.check:
        for scene_group in scene_groups:
            scene = load_scene_data(
                scene_group, args.max_grasps, args.sample_seed
            )
            model = AllGraspsModel(scene, args.hand_xml, args.hand_alpha)
            model.set_phase(args.phase)
            _print_scene_summary(scene, model)
        print("Validation passed.")
        return 0

    print("Keys: 1 pregrasp | 2 grasp | 3 squeeze | H show/hide hands")
    print("      Right/D next scene | Left/A previous scene | Q/Esc quit")
    index = 0
    while 0 <= index < len(scene_groups):
        try:
            scene = load_scene_data(
                scene_groups[index], args.max_grasps, args.sample_seed
            )
            delta, should_quit = show_scene(scene, index, len(scene_groups), args)
        except Exception as exc:
            print(
                f"Skip {scene_groups[index].name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            delta, should_quit = 1, False
        if should_quit:
            break
        index = max(0, index + delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
