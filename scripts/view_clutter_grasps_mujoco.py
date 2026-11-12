#!/usr/bin/env python3
"""Inspect single-object and scene grasps from the generated dataset catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

import mujoco
import numpy as np
from ruamel.yaml import YAML

from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.mujoco_util import delete_spec_element


PHASES = ("pregrasp", "grasp", "squeeze")
REPO_ROOT = Path(__file__).resolve().parents[1]
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
    samples: list[tuple[str, Path]] | None = None


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


def parse_args(argv=None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    preliminary_args, _ = config_parser.parse_known_args(argv)
    config = _load_viewer_config(preliminary_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Inspect single-object or scene grasps from output/dataset_index.json."
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
        help="Explicit succ_collect/filtered_grasps directory; overrides --dataset.",
    )
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "output/dataset_index.json")
    parser.add_argument("--dataset", default=None,
                        help="Dataset ID; defaults to the deduplicated GraspNet test library when available.")
    parser.add_argument("--object", default="*", help="Single-object ID or glob; scales stay separate.")
    parser.add_argument(
        "--taxonomy",
        default="all",
        help="Taxonomy folder to load; use 'all' to combine every taxonomy.",
    )
    parser.add_argument(
        "--scene",
        "--item",
        default="",
        help="Only load one scene or exact object/scale item shown by the panel.",
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
            "downsampling (see --sampling); "
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
        "--sampling", choices=("fps", "random"), default="fps",
        help="Downsampling method, applied independently per target object.",
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
        "--show-contacts", action="store_true",
        help="Replay target contacts: green allowed, red outside the selected taxonomy.",
    )
    parser.add_argument(
        "--render-dir", type=Path,
        help="Export a PNG of each selected scene without opening a window.",
    )
    parser.add_argument(
        "--display",
        help="Explicit X11 display, e.g. :1; overrides desktop auto-detection.",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Open the dataset, grasp-type and sampling control panel.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compile selected scenes and validate qposes without opening a window.",
    )
    args = parser.parse_args(argv)
    if args.root is None:
        try:
            catalog = json.loads(args.catalog.read_text())
            entries = {entry["id"]: entry for entry in catalog["datasets"]}
            if args.dataset is None:
                args.dataset = next((name for name in (
                    'deduplicated_graspnet_test_clutter', 'taxonomy_filtered_graspnet_test_clutter',
                    'graspnet_test_clutter',
                ) if name in entries), next(iter(entries), 'graspnet_test_clutter'))
            if args.dataset not in entries:
                parser.error(f"Unknown dataset {args.dataset!r}; choose from {', '.join(entries)}")
            args.root = str((args.catalog.parent / entries[args.dataset]["data"]).resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Cannot read dataset catalog {args.catalog}: {exc}")
    elif args.dataset is None:
        args.dataset = 'graspnet_test_clutter'
    return args


def configure_display(requested: str | None = None) -> None:
    """Prefer the user's local X server over an inherited login-screen display."""
    current = os.environ.get("DISPLAY", "")
    selected = requested if requested is not None else current
    local = re.fullmatch(r"(?:unix)?:([0-9]+)(?:\.[0-9]+)?", current)
    if requested is None and sys.platform == "linux" and (not current or local):
        owned_displays = set()
        for path in Path("/tmp/.X11-unix").glob("X[0-9]*"):
            try:
                info = path.stat()
            except OSError:
                continue
            if (path.name[1:].isdigit() and info.st_uid == os.getuid()
                    and stat.S_ISSOCK(info.st_mode)):
                owned_displays.add(int(path.name[1:]))
        current_number = int(local.group(1)) if local else None
        if current_number not in owned_displays and len(owned_displays) == 1:
            selected = f":{next(iter(owned_displays))}"
            print(
                f"Desktop display: {current or '<unset>'} -> {selected} "
                "(X11 socket owned by the current user).",
                flush=True,
            )
    if selected:
        os.environ["DISPLAY"] = selected
    elif sys.platform == "linux":
        raise RuntimeError(
            "No X11 DISPLAY found. Set --display to your desktop's display; "
            "use --check for validation without a window."
        )
    print(f"Graphics display: DISPLAY={selected or '<system default>'}", flush=True)


def _scene_selector(value: str) -> str:
    if value.isdigit():
        return f"scene_{int(value):06d}"
    return value


def collect_scene_groups(args: argparse.Namespace) -> tuple[list[SceneGroup], int]:
    root = Path(args.root)
    if not root.is_dir():
        candidates = sorted((REPO_ROOT / "output").glob("*/runs/*/filtered_grasps"))
        hint = ""
        if candidates:
            hint = "\nAvailable filtered grasp roots:\n  " + "\n  ".join(
                str(path) for path in candidates
            )
        raise FileNotFoundError(
            f"Filtered grasp root does not exist: {root}{hint}"
        )

    if args.taxonomy.lower() == "all":
        taxonomy_dirs = sorted(path for path in root.glob("[0-9]*") if path.is_dir())
    else:
        taxonomy_dir = root / args.taxonomy
        if not taxonomy_dir.is_dir():
            raise FileNotFoundError(
                f"Taxonomy directory does not exist: {taxonomy_dir}"
            )
        taxonomy_dirs = [taxonomy_dir]

    grouped: dict[str, list[Path]] = {}
    isolated: dict[str, list[tuple[str, Path]]] = {}
    for taxonomy_dir in taxonomy_dirs:
        for directory in sorted(taxonomy_dir.iterdir()):
            if not directory.is_dir():
                continue
            if directory.name.startswith("scene_"):
                if any(directory.glob("*.npy")):
                    grouped.setdefault(directory.name, []).append(directory)
            elif fnmatch.fnmatch(directory.name, getattr(args, "object", "*")):
                for sample in sorted(directory.rglob("*.npy")):
                    # Include the scale filename so different physical scales
                    # are never combined into the same displayed object.
                    name = str(sample.relative_to(taxonomy_dir).with_suffix(""))
                    isolated.setdefault(name, []).append((taxonomy_dir.name, sample))
    scene_groups = [
        SceneGroup(name=name, directories=directories)
        for name, directories in sorted(grouped.items())
    ] + [SceneGroup(name=name, directories=[], samples=samples)
         for name, samples in sorted(isolated.items())]
    if args.scene:
        requested = args.scene
        names = {requested, _scene_selector(requested)}
        if requested.isdigit():
            names.add(f"scene_{int(requested):04d}")
        scene_groups = [group for group in scene_groups if group.name in names]
        if not scene_groups:
            available = ", ".join(sorted(grouped)[:20]) or "<none>"
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
    records: list[GraspRecord], max_grasps: int, seed: int, sampling: str = "fps"
) -> list[GraspRecord]:
    """Sample independently for every target object in the scene."""
    if sampling not in ("fps", "random"):
        raise ValueError(f"Unknown sampling method: {sampling}")
    if max_grasps <= 0:
        return records

    grouped: dict[str, list[GraspRecord]] = {}
    for record in records:
        grouped.setdefault(record.target_name, []).append(record)

    sampled: list[GraspRecord] = []
    for target_index, target_name in enumerate(sorted(grouped)):
        candidates = grouped[target_name]
        if sampling == "random" and len(candidates) > max_grasps:
            indices = np.random.default_rng(seed + target_index).choice(
                len(candidates), size=max_grasps, replace=False
            )
            sampled.extend(candidates[int(index)] for index in indices)
        else:
            sampled.extend(_farthest_point_downsample(
                candidates, max_grasps, seed + target_index
            ))
    return sampled


def load_scene_data(
    scene_group: SceneGroup, max_grasps: int, sample_seed: int = 12,
    sampling: str = "fps",
) -> SceneData:
    records: list[GraspRecord] = []
    scene_path: Path | None = None

    samples = scene_group.samples
    if samples is None:
        samples = [(directory.parent.name, path)
                   for directory in scene_group.directories
                   for path in sorted(directory.glob("*.npy"))]
    for taxonomy, sample_path in samples:
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

        shapes = [arrays[phase].shape for phase in PHASES]
        if (any(value.ndim != 2 or value.shape[1] != 29 for value in arrays.values())
                or len(set(shapes)) != 1
                or any(not np.isfinite(value).all() for value in arrays.values())):
            raise ValueError(f"Invalid or mismatched grasp phases in {sample_path}: {shapes}")
        count = len(arrays["grasp"])
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
        records, max_grasps, sample_seed + scene_number, sampling
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
                    delete_spec_element(child_spec, geom)
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
            delete_spec_element(child_spec, actuator)

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
        # Hands are visual-only; the scene owns the global solver options.
        child_spec.option.cone = self.spec.option.cone
        child_spec.option.impratio = self.spec.option.impratio
        joint_info = [(joint.name, _qpos_width(joint.type)) for joint in child_spec.joints]
        for actuator in list(child_spec.actuators):
            delete_spec_element(child_spec, actuator)
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


class ContactOverlay:
    """Show real target contacts against the original taxonomy's body rules."""

    def __init__(self, scene: SceneData, model: AllGraspsModel, hand_xml: str):
        from dexonomy.util.taxonomy_contacts import ContactReplay, ContactRules

        self.scene, self.visual = scene, model
        self.rules = ContactRules.load(REPO_ROOT / "assets/hand/shadow/init_tmpl")
        self.replay = ContactReplay(scene.scene_cfg, hand_xml, self.rules, distance=.002)
        self.points = []
        self.summaries = []
        targets = {record.target_name for record in scene.grasps}
        self.focus_center = np.mean([np.asarray(scene.scene_cfg["scene"][target]["pose"][:3])
                                     for target in targets], axis=0)
        self.focus_center[2] = max(float(self.focus_center[2]), .05)

    @staticmethod
    def missing_group_gap(replay, names: list[str]) -> dict:
        """Nearest collision surfaces for a missing group, independent of margin."""
        model, data = replay.model, replay.data
        collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        target_geoms = np.flatnonzero(replay.target & collision)
        hand_bodies = {model.body("hand-" + name).id for name in names}
        hand_geoms = np.flatnonzero(np.isin(replay.geom_body, list(hand_bodies)) & collision)
        best, segment, nearest = np.inf, None, None
        for hand in hand_geoms:
            for target in target_geoms:
                endpoints = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, data, int(hand), int(target), 1., endpoints)
                if distance < best and np.any(endpoints):
                    best, segment = float(distance), endpoints.tolist()
                    nearest = model.body(int(replay.geom_body[hand])).name.removeprefix("hand-")
        return {"bodies":names, "nearest_body":nearest,
                "distance_m":best if np.isfinite(best) else None, "segment_m":segment}

    def update(self, phase: str) -> None:
        self.points, self.summaries = [], []
        model = self.visual.model
        for index, record in enumerate(self.scene.grasps):
            taxonomy = self.rules.names.index(record.taxonomy)
            self.replay.set_target(record.target_name)
            result = self.replay.query(record.qposes[phase], taxonomy, diagnostic=False)
            permitted = set(self.rules.templates[taxonomy]["hand_cbody"])
            missing_groups = [group for i,group in enumerate(self.rules.templates[taxonomy]["required_cbody"])
                              if result["missing"] & (1 << i)]
            missing_bodies = {name for group in missing_groups for name in group}
            actual, extra = set(), set()
            descriptions = []
            for contact in self.replay.data.contact:
                first, second = int(contact.geom1), int(contact.geom2)
                if self.replay.geom_hand_bits[first] and self.replay.target[second]:
                    hand_geom = first
                elif self.replay.geom_hand_bits[second] and self.replay.target[first]:
                    hand_geom = second
                else:
                    continue
                distance = float(contact.dist)
                if distance > self.replay.distance:
                    continue
                body = self.replay.model.body(int(self.replay.geom_body[hand_geom])).name.removeprefix("hand-")
                unexpected = body not in permitted
                actual.add(body)
                if unexpected:
                    extra.add(body)
                    descriptions.append(f"{body.removeprefix('rh_')}: {distance*1000:.2f} mm")
                color = [1., .12, .12, 1.] if unexpected else [.05, .85, .40, 1.]
                self.points.append((np.asarray(contact.pos).copy(), color))
            observed_mask = self.rules.body_mask(actual)
            if observed_mask != result["bodies"]:
                raise RuntimeError("Contact overlay and replay classification disagree")
            prefix = f"grasp-{index:05d}-"
            for geom_id in range(model.ngeom):
                body_name = model.body(int(model.geom_bodyid[geom_id])).name
                if not body_name.startswith(prefix):
                    continue
                name = body_name.removeprefix(prefix)
                model.geom_matid[geom_id] = -1
                model.geom_rgba[geom_id] = (
                    [1., .77, .08, .95] if name in missing_bodies else
                    [1., .18, .13, .95] if name in extra else
                    [.13, .65, .43, .7] if name in actual else [.72, .77, .82, .40]
                )
            self.summaries.append({
                "taxonomy": record.taxonomy, "target": record.target_name,
                "allowed_bodies": sorted(permitted), "actual_bodies": sorted(actual),
                "extra_bodies": sorted(extra), "extra_contact_distances": descriptions,
                "missing_required_groups": missing_groups,
                "missing_group_gaps": [self.missing_group_gap(self.replay, group) for group in missing_groups],
                "matching_taxonomies": [name for i,name in enumerate(self.rules.names)
                                         if result["matches"] & (1 << i)],
            })

    def append_geoms(self, scene: mujoco.MjvScene) -> None:
        for position, color in self.points:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.0035, .0035, .0035]), position,
                               np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1

    def viewer_text(self) -> str:
        lines = ["TARGET CONTACT AUDIT (<= 2 mm)",
                 "RED: extra body/contact | GREEN: allowed contact",
                 "YELLOW: missing required contact group",
                 "Contact markers enlarged for visibility", ""]
        for item in self.summaries[:4]:
            lines.append(f"{item['taxonomy']} | object {item['target']}")
            lines.append("Allowed: " + ", ".join(name.removeprefix('rh_') for name in item['allowed_bodies']))
            lines.append("Extra: " + ("; ".join(item['extra_contact_distances']) or "none"))
            for gap in item["missing_group_gaps"]:
                group = " or ".join(name.removeprefix('rh_') for name in gap['bodies'])
                distance = gap['distance_m']
                value = f"{distance*1000:.6f} mm" if distance is not None else "unavailable"
                lines.append(f"Missing: {group} | nearest gap {value}")
        lines.append("Left/Right: example | H: hide hands | Q: exit")
        return "\n".join(lines)


def render_scene(scene: SceneData, args: argparse.Namespace) -> None:
    import imageio.v3 as iio

    visual = AllGraspsModel(scene, args.hand_xml, args.hand_alpha)
    visual.set_phase(args.phase)
    overlay = ContactOverlay(scene, visual, args.hand_xml) if args.show_contacts else None
    if overlay:
        overlay.update(args.phase)
    visual.model.vis.global_.offwidth = 1200
    visual.model.vis.global_.offheight = 900
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = overlay.focus_center if overlay else visual.center
    camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
    with mujoco.Renderer(visual.model, height=900, width=1200) as renderer:
        renderer.update_scene(visual.data, camera=camera)
        if overlay:
            overlay.append_geoms(renderer.scene)
        frame = renderer.render()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    destination = args.render_dir / (scene.name.replace("/", "__") + ".png")
    iio.imwrite(destination, frame)
    if overlay:
        destination.with_suffix(".json").write_text(json.dumps(overlay.summaries, indent=2) + "\n")
    print(f"Rendered {destination}", flush=True)


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
    overlay = ContactOverlay(scene, all_grasps, args.hand_xml) if args.show_contacts else None
    if overlay:
        overlay.update(args.phase)

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
        show_left_ui=not args.show_contacts,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = overlay.focus_center if overlay else all_grasps.center
        viewer.cam.distance = args.distance
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        if overlay:
            overlay.append_geoms(viewer.user_scn)
            viewer.set_texts((None, None, overlay.viewer_text(), ""))

        while viewer.is_running():
            if state["quit"]:
                return 0, True
            if state["scene_delta"]:
                return int(state["scene_delta"]), False
            if state["phase"] != all_grasps.phase:
                all_grasps.set_phase(str(state["phase"]))
                if overlay:
                    overlay.update(all_grasps.phase)
                    viewer.user_scn.ngeom = 0
                    overlay.append_geoms(viewer.user_scn)
                    viewer.set_texts((None, None, overlay.viewer_text(), ""))
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

    if args.gui:
        try:
            configure_display(args.display)
            if __package__:
                from .grasp_viewer_panel import launch_panel
            else:
                from grasp_viewer_panel import launch_panel
            return launch_panel(args, collect_scene_groups)
        except (ImportError, RuntimeError) as exc:
            print(f"Cannot open viewer panel: {exc}", file=sys.stderr)
            return 1

    try:
        scene_groups, total = collect_scene_groups(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if not scene_groups:
        print("No matching grasp scenes or object scales found.", file=sys.stderr)
        return 1

    print(f"Found {total} matching scenes; selected {len(scene_groups)}.")
    if args.render_dir:
        for group in scene_groups:
            render_scene(load_scene_data(group, args.max_grasps, args.sample_seed, args.sampling), args)
        return 0
    if args.check:
        for scene_group in scene_groups:
            scene = load_scene_data(
                scene_group, args.max_grasps, args.sample_seed, args.sampling
            )
            model = AllGraspsModel(scene, args.hand_xml, args.hand_alpha)
            model.set_phase(args.phase)
            _print_scene_summary(scene, model)
        print("Validation passed.")
        return 0

    try:
        configure_display(args.display)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Keys: 1 pregrasp | 2 grasp | 3 squeeze | H show/hide hands")
    print("      Right/D next scene | Left/A previous scene | Q/Esc quit")
    index = 0
    while 0 <= index < len(scene_groups):
        try:
            scene = load_scene_data(
                scene_groups[index], args.max_grasps, args.sample_seed, args.sampling
            )
            delta, should_quit = show_scene(scene, index, len(scene_groups), args)
        except Exception as exc:
            print(
                f"Cannot display {scene_groups[index].name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
        if should_quit:
            break
        index = max(0, index + delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
