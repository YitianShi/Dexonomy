"""Generate MuJoCo clutter scenes and transfer/filter single-object grasps."""

from __future__ import annotations

import copy
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from glob import glob
import json
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import shutil
from tempfile import TemporaryDirectory
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from omegaconf import OmegaConf

from dexonomy.sim.clutter_env import ClutterSimCfg, MuJoCoClutterEnv, sample_spawn_poses
from dexonomy.util.file_util import load_json, load_scene_cfg

if TYPE_CHECKING:
    from dexonomy.sim import MuJoCo_EvalEnv, MuJoCo_OptEnv


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig":
        return [_plain(item) for item in value]
    return value


def _scene_sim_cfg(cfg) -> ClutterSimCfg:
    return ClutterSimCfg(
        timestep=float(cfg.timestep),
        table_size=tuple(cfg.table_size),
        table_height=float(cfg.table_height),
        table_thickness=float(cfg.table_thickness),
        friction=float(cfg.friction),
        table_friction=float(cfg.table_friction),
        impratio=float(cfg.impratio),
        noslip_iterations=int(cfg.noslip_iterations),
        object_mass=float(cfg.object_mass),
        spawn_z=tuple(cfg.spawn_z),
        spawn_xy_scale=float(cfg.spawn_xy_scale),
        spawn_gap=float(cfg.spawn_gap),
        spawn_position_attempts=int(cfg.spawn_position_attempts),
        max_object_speed=float(cfg.max_object_speed),
        freejoint_damping=float(cfg.freejoint_damping),
        containment_walls=bool(cfg.containment_walls),
        containment_height=float(cfg.containment_height),
        containment_thickness=float(cfg.containment_thickness),
        bound_mass=float(cfg.bound_mass),
        bound_inertia=float(cfg.bound_inertia),
        debug_view=bool(cfg.debug_view),
        debug_realtime=bool(cfg.debug_realtime),
        debug_spawn_pause=float(cfg.debug_spawn_pause),
        debug_result_pause=float(cfg.debug_result_pause),
        settle_timeout=float(cfg.settle_timeout),
        stable_time=float(cfg.stable_time),
        linear_velocity=float(cfg.linear_velocity),
        angular_velocity=float(cfg.angular_velocity),
        penetration_tolerance=float(cfg.penetration_tolerance),
        table_penetration_tolerance=float(cfg.table_penetration_tolerance),
    )


def _object_density(obj_cfg: dict, object_mass: float) -> float:
    info = load_json(obj_cfg["info_path"])
    source_density = float(info["density"])
    source_scale = float(info.get("scale", 1.0))
    unscaled_mass = float(info["mass"]) / (source_density * source_scale**3)
    return object_mass / (unscaled_mass * float(np.prod(obj_cfg["scale"])))


def _build_catalog(scene_cfg_glob: str, cfg) -> dict[str, list[dict]]:
    paths = sorted(glob(scene_cfg_glob, recursive=True))
    if cfg.catalog_limit and int(cfg.catalog_limit) > 0:
        paths = paths[: int(cfg.catalog_limit)]
    catalog: dict[str, list[dict]] = {}
    info_cache: dict[str, dict] = {}
    for path in paths:
        try:
            scene_cfg = load_scene_cfg(path)
            name = str(scene_cfg["task"]["obj_name"])
            obj_cfg = scene_cfg["scene"][name]
            if obj_cfg["type"] != "rigid_object":
                continue
            for key, value in list(obj_cfg.items()):
                if key.endswith("_path") and isinstance(value, str):
                    obj_cfg[key] = os.path.abspath(value)
            info_path = obj_cfg["info_path"]
            if info_path not in info_cache:
                info_cache[info_path] = load_json(info_path)
            extent = np.asarray(info_cache[info_path]["obb"], dtype=float) * np.asarray(
                obj_cfg["scale"], dtype=float
            )
            max_extent = float(extent.max())
            if max_extent < float(cfg.min_extent) or max_extent > float(cfg.max_extent):
                continue
            catalog.setdefault(name, []).append(
                {
                    "name": name,
                    "scene_path": os.path.abspath(path),
                    "cfg": copy.deepcopy(obj_cfg),
                    "raw_extent": np.asarray(
                        info_cache[info_path]["obb"], dtype=float
                    ),
                    "extent": extent,
                }
            )
        except (KeyError, OSError, ValueError) as exc:
            logging.debug("Skip asset %s: %s", path, exc)
    logging.info(
        "Clutter asset catalog: %d objects, %d scale variants",
        len(catalog),
        sum(len(variants) for variants in catalog.values()),
    )
    return catalog


def _dataset_scale_choices(
    name: str,
    cfg,
    cache: dict[str, list[dict]],
) -> list[dict]:
    """Load per-grasp physical scales lazily for one object."""
    if name in cache:
        return cache[name]

    grasp_root = Path(cfg.grasp_scale_dir)
    choices = []
    required = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
    for taxonomy in _taxonomy_names(grasp_root, cfg.grasp_scale_taxonomies):
        variant_dir = grasp_root / taxonomy / name / "floating"
        for grasp_path in sorted(variant_dir.glob("*.npy")):
            try:
                grasp_data = np.load(grasp_path, allow_pickle=True).item()
                if any(key not in grasp_data for key in required):
                    continue
                source_scene_path = os.path.abspath(str(grasp_data["scene_path"]))
                source_scene = load_scene_cfg(source_scene_path)
                source_obj_cfg = source_scene["scene"][name]
                if source_obj_cfg["type"] != "rigid_object":
                    continue
                for key, value in list(source_obj_cfg.items()):
                    if key.endswith("_path") and isinstance(value, str):
                        source_obj_cfg[key] = os.path.abspath(value)
                source_scale = np.asarray(source_obj_cfg["scale"], dtype=np.float64)
                raw_extent = np.asarray(
                    load_json(source_obj_cfg["info_path"])["obb"], dtype=np.float64
                )
                count = min(len(np.atleast_2d(grasp_data[key])) for key in required)
                for index in range(count):
                    multiplier = _scene_scale_multiplier(grasp_data, index, count)
                    effective_scale = source_scale * multiplier
                    extent = raw_extent * effective_scale
                    max_extent = float(extent.max())
                    if (
                        max_extent < float(cfg.min_extent)
                        or max_extent > float(cfg.max_extent)
                    ):
                        continue
                    object_cfg = copy.deepcopy(source_obj_cfg)
                    object_cfg["scale"] = effective_scale
                    choices.append(
                        {
                            "name": name,
                            "scene_path": source_scene_path,
                            "cfg": object_cfg,
                            "raw_extent": raw_extent,
                            "extent": extent,
                            "grasp_scale_source": {
                                "taxonomy": taxonomy,
                                "source_grasp_path": os.path.abspath(grasp_path),
                                "source_index": index,
                                "source_object_scale": source_scale,
                                "scene_scale": multiplier,
                                "effective_object_scale": effective_scale,
                            },
                        }
                    )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                logging.debug("Skip grasp scale source %s: %s", grasp_path, exc)
    cache[name] = choices
    return choices


def _sample_dataset_scale_assets(
    catalog: dict[str, list[dict]],
    object_names: list[str],
    count: int,
    cfg,
    cache: dict[str, list[dict]],
    rng: np.random.Generator,
) -> list[dict]:
    """Sample distinct objects and one released grasp scale for each."""
    assets = []
    for name_index in rng.permutation(len(object_names)):
        name = object_names[int(name_index)]
        if name not in catalog:
            continue
        choices = _dataset_scale_choices(name, cfg, cache)
        if not choices:
            continue
        assets.append(copy.deepcopy(choices[int(rng.integers(len(choices)))]))
        if len(assets) >= count:
            break
    return assets


def _runtime_objects(assets: list[dict], sim_cfg: ClutterSimCfg) -> list[dict]:
    objects = []
    for asset in assets:
        obj_cfg = asset["cfg"]
        objects.append(
            {
                "name": asset["name"],
                "xml_path": obj_cfg["xml_path"],
                "scale": np.asarray(obj_cfg["scale"], dtype=float),
                "density": _object_density(obj_cfg, sim_cfg.object_mass),
                "extent": np.asarray(asset["extent"], dtype=float),
            }
        )
    return objects


def _relative_object_cfg(obj_cfg: dict, output_path: Path, pose: np.ndarray) -> dict:
    result = copy.deepcopy(obj_cfg)
    result["pose"] = np.asarray(pose, dtype=np.float32)
    for key, value in list(result.items()):
        if key.endswith("_path") and isinstance(value, str):
            result[key] = os.path.relpath(os.path.abspath(value), output_path.parent)
    return result


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(os.sep, "__")


def _save_scene_targets(
    scene_index: int,
    assets: list[dict],
    poses: np.ndarray,
    settle_info: dict,
    cfg,
    sim_cfg: ClutterSimCfg,
    rng: np.random.Generator,
) -> list[str]:
    output_dir = Path(cfg.output_dir) / f"scene_{scene_index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    n_targets = int(cfg.targets_per_scene)
    if n_targets <= 0 or n_targets >= len(assets):
        target_indices = np.arange(len(assets))
    else:
        target_indices = np.sort(rng.choice(len(assets), n_targets, replace=False))

    saved_paths = []
    for target_index in target_indices:
        target_name = assets[int(target_index)]["name"]
        output_path = output_dir / f"{_safe_name(target_name)}.npy"
        scene = {
            asset["name"]: _relative_object_cfg(asset["cfg"], output_path, pose)
            for asset, pose in zip(assets, poses)
        }
        scene["table"] = {
            "type": "plane",
            "friction": np.asarray(
                [sim_cfg.table_friction, 0.005, 0.0001], dtype=np.float64
            ),
            "pose": np.asarray([0, 0, sim_cfg.table_height, 1, 0, 0, 0], dtype=np.float32),
            "size": np.asarray(
                [sim_cfg.table_size[0] / 2, sim_cfg.table_size[1] / 2, 0.01],
                dtype=np.float32,
            ),
        }
        source_paths = {
            asset["name"]: os.path.relpath(asset["scene_path"], os.getcwd())
            for asset in assets
        }
        selected_grasp_scales = {
            asset["name"]: {
                key: (
                    os.path.relpath(value, os.getcwd())
                    if key == "source_grasp_path"
                    else value
                )
                for key, value in asset["grasp_scale_source"].items()
            }
            for asset in assets
            if "grasp_scale_source" in asset
        }
        data = {
            "scene": scene,
            "scene_id": f"clutter/scene_{scene_index:06d}/{target_name}",
            "task": {"type": "force_closure", "obj_name": target_name},
            "clutter": {
                "generator": "mujoco",
                "source_scene_paths": source_paths,
                "selected_grasp_scales": selected_grasp_scales,
                "object_names": [asset["name"] for asset in assets],
                "settle": settle_info,
                "sim": asdict(sim_cfg),
            },
        }
        np.save(output_path, data)
        saved_paths.append(str(output_path))

    # A regenerated scene can have different targets. Keep only this run's
    # configs, so later filtering cannot load stale versions of the scene.
    for existing_path in output_dir.glob("*.npy"):
        if str(existing_path) not in saved_paths:
            existing_path.unlink()
    return saved_paths


def _generate_scene(
    scene_index: int,
    cfg,
    sim_cfg: ClutterSimCfg,
    catalog: dict[str, list[dict]],
    object_names: list[str],
    grasp_scale_cache: dict[str, list[dict]],
    rng: np.random.Generator,
) -> tuple[list[str], dict]:
    """Attempt one scene ID; return a rejection for table penetration."""
    last_info = {"reason": "not_started"}
    scene_failures: Counter[str] = Counter()
    for scene_attempt in range(int(cfg.max_scene_attempts)):
        attempt_failures: Counter[str] = Counter()
        n_objects = int(rng.integers(int(cfg.objects_min), int(cfg.objects_max) + 1))
        if bool(cfg.sample_dataset_scales):
            assets = _sample_dataset_scale_assets(
                catalog,
                object_names,
                n_objects,
                cfg,
                grasp_scale_cache,
                rng,
            )
            if len(assets) < n_objects:
                last_info = {
                    "reason": "insufficient_grasp_scale_assets",
                    "requested": n_objects,
                    "found": len(assets),
                }
                scene_failures["insufficient_grasp_scale_assets"] += 1
                continue
            chosen_names = [asset["name"] for asset in assets]
        else:
            chosen_names = [
                str(name)
                for name in rng.choice(object_names, n_objects, replace=False)
            ]
            assets = [
                copy.deepcopy(catalog[name][int(rng.integers(len(catalog[name])))])
                for name in chosen_names
            ]
        runtime_objects = _runtime_objects(assets, sim_cfg)
        if sim_cfg.debug_view:
            logging.info(
                "Debug scene %06d asset attempt %d/%d with objects=%s",
                scene_index,
                scene_attempt + 1,
                int(cfg.max_scene_attempts),
                ",".join(chosen_names),
            )
        try:
            env = MuJoCoClutterEnv(runtime_objects, sim_cfg)
        except Exception as exc:
            last_info = {"reason": "model_compile", "error": str(exc)}
            attempt_failures["model_compile"] += 1
            scene_failures.update(attempt_failures)
            logging.debug("Scene %06d compile retry: %s", scene_index, exc)
            continue
        try:
            for spawn_attempt in range(int(cfg.max_spawn_attempts)):
                if sim_cfg.debug_view:
                    logging.info(
                        "Debug scene %06d spawn attempt %d/%d",
                        scene_index,
                        spawn_attempt + 1,
                        int(cfg.max_spawn_attempts),
                    )
                spawn_poses = sample_spawn_poses(runtime_objects, sim_cfg, rng)
                if not env.set_spawn_poses(spawn_poses):
                    penetration = env.get_table_penetration()
                    if penetration is not None:
                        last_info = penetration
                        attempt_failures["table_penetration"] += 1
                        break
                    last_info = {"reason": "initial_penetration"}
                    attempt_failures["initial_penetration"] += 1
                    continue
                stable, poses, last_info = env.settle()
                if not stable:
                    attempt_failures[str(last_info.get("reason", "unknown"))] += 1
                    if last_info.get("reason") == "table_penetration":
                        break
                    if last_info.get("reason") in {
                        "simulation_instability",
                        "nonfinite_state",
                    }:
                        logging.warning(
                            "Reject unstable scene %06d (asset attempt %d/%d, "
                            "spawn attempt %d/%d, objects=%s): %s",
                            scene_index,
                            scene_attempt + 1,
                            int(cfg.max_scene_attempts),
                            spawn_attempt + 1,
                            int(cfg.max_spawn_attempts),
                            ",".join(chosen_names),
                            last_info,
                        )
                    continue
                saved = _save_scene_targets(
                    scene_index, assets, poses, last_info, cfg, sim_cfg, rng
                )
                record = {
                    "scene_index": scene_index,
                    "object_names": list(chosen_names),
                    "target_paths": saved,
                    "settle": last_info,
                }
                logging.info(
                    "Generated scene %06d with %d objects and %d targets",
                    scene_index,
                    n_objects,
                    len(saved),
                )
                if bool(cfg.sample_dataset_scales):
                    logging.info(
                        "Scene %06d sampled dataset scales: %s",
                        scene_index,
                        ", ".join(
                            f"{asset['name']}={float(np.asarray(asset['cfg']['scale'])[0]):.5f}"
                            for asset in assets
                        ),
                    )
                return saved, record
        finally:
            env.close()
        scene_failures.update(attempt_failures)
        if last_info.get("reason") == "table_penetration":
            record = {
                "scene_index": scene_index,
                "status": "rejected",
                "reason": "table_penetration",
                "object_names": list(chosen_names),
                "target_paths": [],
                "settle": last_info,
                "failures": dict(scene_failures),
            }
            logging.warning(
                "Reject scene %06d after table penetration: %s",
                scene_index,
                last_info,
            )
            return [], record
        logging.info(
            "Retry scene %06d after asset attempt %d/%d: failures=%s",
            scene_index,
            scene_attempt + 1,
            int(cfg.max_scene_attempts),
            dict(attempt_failures),
        )
    raise RuntimeError(
        f"Failed to generate scene {scene_index:06d} after "
        f"{cfg.max_scene_attempts} asset attempts: failures={dict(scene_failures)}, "
        f"last={last_info}"
    )


def _generate_scene_until_valid(
    scene_index: int,
    cfg,
    sim_cfg: ClutterSimCfg,
    catalog: dict[str, list[dict]],
    object_names: list[str],
    grasp_scale_cache: dict[str, list[dict]],
    rng: np.random.Generator,
) -> tuple[list[str], dict]:
    """Resample rejected candidates while retaining the requested scene ID."""
    regeneration_failures: Counter[str] = Counter()
    while True:
        paths, record = _generate_scene(
            scene_index, cfg, sim_cfg, catalog, object_names, grasp_scale_cache, rng
        )
        if record.get("status") != "rejected":
            if regeneration_failures:
                record["regeneration_count"] = sum(regeneration_failures.values())
                record["regeneration_failures"] = dict(regeneration_failures)
            return paths, record
        reason = record["reason"]
        regeneration_failures[reason] += 1
        logging.info(
            "Regenerate scene %06d with newly sampled objects (regeneration %d, reason=%s)",
            scene_index,
            sum(regeneration_failures.values()),
            reason,
        )


_SCENE_WORKER_STATE = None


def _init_scene_worker(cfg, sim_cfg: ClutterSimCfg, catalog_path: str, log_level: int, filter_cfg=None):
    global _SCENE_WORKER_STATE
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(processName)s][%(levelname)s] - %(message)s",
        force=True,
    )
    logging.getLogger("OpenGL").setLevel(logging.WARNING)
    with open(catalog_path, "rb") as file:
        catalog = pickle.load(file)
    _SCENE_WORKER_STATE = (cfg, sim_cfg, catalog, sorted(catalog), {}, filter_cfg)
    logging.info("Clutter scene worker ready (pid=%d)", os.getpid())


@contextmanager
def _scene_worker_environment():
    """Keep native math libraries from creating a thread pool in every worker."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scene_process_context():
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
        # A fresh server loads the CPU-only dependencies once, then shares them.
        context.set_forkserver_preload([__name__])
        return context
    return multiprocessing.get_context("spawn")


@contextmanager
def _scene_worker_pool(cfg, sim_cfg: ClutterSimCfg, catalog: dict, worker_count: int, filter_cfg=None):
    # Resolve Hydra's runtime-dependent paths before leaving the parent process.
    worker_cfg = (
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if OmegaConf.is_config(cfg) else cfg
    )
    worker_filter_cfg = (
        OmegaConf.create(OmegaConf.to_container(filter_cfg, resolve=True))
        if OmegaConf.is_config(filter_cfg) else filter_cfg
    )
    with TemporaryDirectory(prefix="dexonomy-scene-workers-") as worker_dir:
        catalog_path = str(Path(worker_dir) / "catalog.pkl")
        with open(catalog_path, "wb") as file:
            pickle.dump(catalog, file, protocol=pickle.HIGHEST_PROTOCOL)
        with _scene_worker_environment():
            context = _scene_process_context()
            logging.info(
                "Starting clutter workers using %s with one native compute thread each",
                context.get_start_method(),
            )
            with context.Pool(
                processes=worker_count,
                initializer=_init_scene_worker,
                initargs=(worker_cfg, sim_cfg, catalog_path, logging.getLogger().getEffectiveLevel(), worker_filter_cfg),
            ) as pool:
                yield pool


def _scene_generation_inputs(cfg):
    sim_cfg = _scene_sim_cfg(cfg)
    catalog_started = time.perf_counter()
    catalog = _build_catalog(str(cfg.source_cfg_glob), cfg)
    logging.info("Clutter asset catalog prepared in %.2fs", time.perf_counter() - catalog_started)
    object_names = sorted(catalog)
    if len(object_names) < int(cfg.objects_max):
        raise ValueError(
            f"Need at least {cfg.objects_max} eligible objects, found {len(object_names)}"
        )
    return sim_cfg, catalog, object_names


def _generate_scene_worker(params) -> tuple[int, list[str], dict]:
    scene_index, seed = params
    cfg, sim_cfg, catalog, object_names, scale_cache, _ = _SCENE_WORKER_STATE
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_index]))
    paths, record = _generate_scene_until_valid(
        scene_index, cfg, sim_cfg, catalog, object_names, scale_cache, rng
    )
    return scene_index, paths, record


def _generate_and_filter_scene_worker(params):
    scene_index, seed, existing_paths = params
    cfg, sim_cfg, catalog, object_names, scale_cache, filter_cfg = _SCENE_WORKER_STATE
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_index]))
    paths, record, filter_records = _process_clutter_scene(
        scene_index, cfg, sim_cfg, catalog, object_names, scale_cache, rng,
        filter_cfg, existing_paths,
    )
    return scene_index, paths, record, filter_records


def _generate_scenes(
    cfg, rng: np.random.Generator, n_worker: int = 1
) -> tuple[list[str], list[dict]]:
    """Generate independent scene IDs using persistent CPU worker processes."""
    paths_by_index: dict[int, list[str]] = {}
    records_by_index: dict[int, dict] = {}
    pending_indices = []
    start_index = int(cfg.start_index)
    for scene_index in range(start_index, start_index + int(cfg.num_scenes)):
        existing = sorted(
            (Path(cfg.output_dir) / f"scene_{scene_index:06d}").glob("*.npy")
        )
        if existing and bool(cfg.skip_existing):
            logging.info("Scene %06d already exists (%d targets)", scene_index, len(existing))
            paths_by_index[scene_index] = [str(path) for path in existing]
        else:
            pending_indices.append(scene_index)
    if not pending_indices:
        return [path for index in sorted(paths_by_index) for path in paths_by_index[index]], []

    sim_cfg, catalog, object_names = _scene_generation_inputs(cfg)
    # Derive each scene's random stream from its ID, independent of scheduling
    # and of which other scene directories already exist during resume.
    scene_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    worker_count = max(1, min(int(n_worker), len(pending_indices)))
    if sim_cfg.debug_view and worker_count > 1:
        logging.info("Debug viewer enabled; generate clutter scenes in one process")
        worker_count = 1
    logging.info(
        "Generate %d clutter scenes with %d CPU workers", len(pending_indices), worker_count
    )
    if worker_count == 1:
        scale_cache: dict[str, list[dict]] = {}
        for scene_index in pending_indices:
            scene_rng = np.random.default_rng(
                np.random.SeedSequence([scene_seed, scene_index])
            )
            paths, record = _generate_scene_until_valid(
                scene_index, cfg, sim_cfg, catalog, object_names, scale_cache, scene_rng
            )
            paths_by_index[scene_index] = paths
            records_by_index[scene_index] = record
    else:
        with _scene_worker_pool(cfg, sim_cfg, catalog, worker_count) as pool:
            params = ((scene_index, scene_seed) for scene_index in pending_indices)
            for completed, (scene_index, paths, record) in enumerate(
                pool.imap_unordered(_generate_scene_worker, params, chunksize=1), start=1
            ):
                paths_by_index[scene_index] = paths
                records_by_index[scene_index] = record
                logging.info(
                    "Finished clutter scene %06d (%d/%d): saved, %d targets",
                    scene_index,
                    completed,
                    len(pending_indices),
                    len(paths),
                )
    return (
        sorted(path for paths in paths_by_index.values() for path in paths),
        [records_by_index[index] for index in sorted(records_by_index)],
    )


def _clear_scene_outputs(scene_dir: Path, filter_output_dir: Path):
    """Remove only a rejected scene and the labels tied to its current geometry."""
    for path in [scene_dir, *filter_output_dir.glob(f"*/{scene_dir.name}")]:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _process_clutter_scene(
    scene_index: int,
    cfg,
    sim_cfg: ClutterSimCfg,
    catalog: dict,
    object_names: list[str],
    scale_cache: dict,
    rng: np.random.Generator,
    filter_cfg,
    existing_paths: list[str] | None = None,
) -> tuple[list[str], dict, list[dict]]:
    """Keep this ID until its scene has at least one filtered grasp."""
    scene_name = f"scene_{scene_index:06d}"
    paths = sorted(existing_paths or [])
    reused_existing = bool(paths)
    scene_dir = Path(cfg.output_dir) / scene_name
    if paths:
        # Filter-only runs can point at a scene root outside cfg.output_dir.
        # Regenerate rejected scenes in their original directory.
        scene_dir = Path(paths[0]).parent
        if scene_dir.name != scene_name or any(Path(path).parent != scene_dir for path in paths):
            raise ValueError(f"Scene {scene_index:06d} has inconsistent target paths: {paths}")
        cfg = copy.deepcopy(cfg)
        cfg.output_dir = str(scene_dir.parent)
        existing_scene = load_scene_cfg(paths[0])
        metadata = existing_scene.get("clutter", {})
        record = {
            "scene_index": scene_index,
            "object_names": metadata.get("object_names", []),
            "target_paths": paths,
            "settle": metadata.get("settle", {}),
        }
        logging.info("Scene %06d already exists; filter its %d targets directly", scene_index, len(paths))

    regeneration_failures: Counter[str] = Counter()
    filter_attempts = 0
    while True:
        if not paths:
            paths, record = _generate_scene_until_valid(
                scene_index, cfg, sim_cfg, catalog, object_names, scale_cache, rng
            )
            regeneration_failures.update(record.get("regeneration_failures", {}))
        filter_attempts += 1
        logging.info("Filter scene %06d immediately (%d targets, attempt %d)", scene_index, len(paths), filter_attempts)
        filter_records = [
            item for scene_path in paths for item in _filter_scene(scene_path, filter_cfg)
        ]
        valid_count = sum(int(item.get("valid_count", 0) or 0) for item in filter_records)
        if valid_count > 0:
            record.update({
                "valid_grasp_count": valid_count,
                "filter_attempts": filter_attempts,
                "reused_existing": reused_existing and filter_attempts == 1,
            })
            if regeneration_failures:
                record["regeneration_count"] = sum(regeneration_failures.values())
                record["regeneration_failures"] = dict(regeneration_failures)
            return paths, record, filter_records

        regeneration_failures["zero_valid_grasps"] += 1
        logging.info(
            "Scene %06d has zero valid grasps; remove its scene and grasp outputs "
            "and resample objects under the same ID (regeneration %d)",
            scene_index, sum(regeneration_failures.values()),
        )
        _clear_scene_outputs(scene_dir, Path(filter_cfg.output_dir))
        paths = []


def _generate_and_filter_scenes(
    op_cfg, rng: np.random.Generator, n_worker: int = 1,
    existing_scene_paths: list[str] | None = None,
) -> tuple[list[str], list[dict], list[dict]]:
    """Schedule complete scene pipelines, including existing IDs on resume."""
    cfg = op_cfg.scene
    tasks: dict[int, list[str]] = {}
    if existing_scene_paths is None:
        start_index = int(cfg.start_index)
        for scene_index in range(start_index, start_index + int(cfg.num_scenes)):
            tasks[scene_index] = sorted(
                str(path) for path in (Path(cfg.output_dir) / f"scene_{scene_index:06d}").glob("*.npy")
            )
    else:
        directories: dict[int, Path] = {}
        for scene_path in existing_scene_paths:
            scene_dir = Path(scene_path).parent
            match = re.fullmatch(r"scene_(\d+)", scene_dir.name)
            if match is None or scene_dir.name != f"scene_{int(match[1]):06d}":
                raise ValueError(f"Expected a scene_XXXXXX directory: {scene_dir}")
            scene_index = int(match[1])
            if scene_index in directories and directories[scene_index] != scene_dir:
                raise ValueError(f"Multiple directories share scene ID {scene_index:06d}")
            directories[scene_index] = scene_dir
        for scene_index, scene_dir in directories.items():
            tasks[scene_index] = sorted(str(path) for path in scene_dir.glob("*.npy"))
    if not tasks:
        return [], [], []

    sim_cfg, catalog, object_names = _scene_generation_inputs(cfg)
    scene_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    worker_count = max(1, min(int(n_worker), len(tasks)))
    if sim_cfg.debug_view:
        worker_count = 1
    logging.info("Generate/reuse and immediately filter %d scenes with %d CPU workers", len(tasks), worker_count)
    results: dict[int, tuple[list[str], dict, list[dict]]] = {}

    def save_result(scene_index, paths, record, filter_records):
        results[scene_index] = (paths, record, filter_records)
        logging.info(
            "Finished clutter scene %06d (%d/%d): %d targets, %d valid grasps",
            scene_index, len(results), len(tasks), len(paths), record["valid_grasp_count"],
        )

    if worker_count == 1:
        scale_cache: dict[str, list[dict]] = {}
        for scene_index in sorted(tasks):
            scene_rng = np.random.default_rng(np.random.SeedSequence([scene_seed, scene_index]))
            save_result(scene_index, *_process_clutter_scene(
                scene_index, cfg, sim_cfg, catalog, object_names, scale_cache,
                scene_rng, op_cfg.filter, tasks[scene_index],
            ))
    else:
        with _scene_worker_pool(cfg, sim_cfg, catalog, worker_count, op_cfg.filter) as pool:
            params = ((index, scene_seed, tasks[index]) for index in sorted(tasks))
            for result in pool.imap_unordered(_generate_and_filter_scene_worker, params, chunksize=1):
                save_result(*result)
    scene_paths = [path for index in sorted(results) for path in results[index][0]]
    scene_records = [results[index][1] for index in sorted(results)]
    filter_records = [record for index in sorted(results) for record in results[index][2]]
    _log_filter_summary(filter_records)
    return scene_paths, scene_records, filter_records


def _transform_qposes(qposes: np.ndarray, old_obj_pose: np.ndarray, new_obj_pose: np.ndarray) -> np.ndarray:
    from dexonomy.util.np_util import np_get_relative_pose, np_multiply_pose

    qposes = np.asarray(qposes, dtype=np.float32)
    if qposes.ndim == 1:
        qposes = qposes[None]
    result = qposes.copy()
    for index, qpos in enumerate(qposes):
        hand_in_object = np_get_relative_pose(old_obj_pose, qpos[:7])
        result[index, :7] = np_multiply_pose(new_obj_pose, hand_in_object)
    return result


def _approach_qposes(pregrasp: np.ndarray, grasp: np.ndarray, steps: int) -> np.ndarray:
    from dexonomy.util.np_util import np_interp_slide

    if steps <= 0:
        return np.stack([pregrasp, grasp])
    roots = np_interp_slide(pregrasp[:7], grasp[:7], steps)
    joints = np.linspace(pregrasp[7:], grasp[7:], steps + 1)[1:]
    return np.concatenate(
        [pregrasp[None], np.concatenate([roots, joints], axis=-1)], axis=0
    )


def _collision_reason(
    env: MuJoCo_OptEnv,
    qpos: np.ndarray,
    cfg,
    *,
    check_target_penetration: bool,
) -> str | None:
    env.reset_qpos(qpos)
    target_contact, scene_contact, self_contact = env.get_scene_contacts(
        float(cfg.contact_margin)
    )
    if len(scene_contact["dist"]) and float(np.min(scene_contact["dist"])) < float(
        cfg.scene_threshold
    ):
        return "scene_collision"
    if bool(cfg.reject_self_collision) and len(self_contact["dist"]):
        if float(np.min(self_contact["dist"])) < float(cfg.self_threshold):
            return "self_collision"
    if check_target_penetration and len(target_contact["dist"]):
        if float(np.min(target_contact["dist"])) < float(
            cfg.target_penetration_threshold
        ):
            return "target_penetration"
    return None


def _has_target_contact(env: MuJoCo_OptEnv, qpos: np.ndarray, margin: float) -> bool:
    env.reset_qpos(qpos)
    target_contact, _, _ = env.get_scene_contacts(margin)
    return len(target_contact["dist"]) > 0


def _scene_with_mass(scene_cfg: dict, object_mass: float) -> dict:
    result = copy.deepcopy(scene_cfg)
    for obj_cfg in result["scene"].values():
        if obj_cfg["type"] == "rigid_object":
            obj_cfg["density"] = _object_density(obj_cfg, object_mass)
    return result


def _passes_lift(
    env: MuJoCo_EvalEnv,
    pregrasp: np.ndarray,
    grasp: np.ndarray,
    squeeze: np.ndarray,
    cfg,
) -> bool:
    env.reset_qpos(pregrasp)
    env.step_sim(int(cfg.settle_substeps))
    initial_pose = env.get_active_obj_pose().copy()
    lift = squeeze.copy()
    lift[2] += float(cfg.height)
    env.run_trajectory(
        [pregrasp, grasp, squeeze, lift],
        ["ee_pose", "ee_pose", "ee_pose"],
        [None, None, None],
        step=int(cfg.steps),
        substep=int(cfg.substeps),
    )
    env.step_sim(int(cfg.hold_substeps))
    final_pose = env.get_active_obj_pose()
    return float(final_pose[2] - initial_pose[2]) >= float(cfg.success_height)


def _taxonomy_names(source_dir: Path, configured) -> list[str]:
    if configured is None:
        return sorted(path.name for path in source_dir.iterdir() if path.is_dir())
    if isinstance(configured, str):
        return [configured]
    return [str(value) for value in configured]


def _scene_scale_multiplier(source_data: dict, index: int, total_count: int) -> np.ndarray:
    """Return the released per-grasp scale as a three-axis multiplier."""
    if "scene_scale" not in source_data:
        return np.ones(3, dtype=np.float64)
    values = np.asarray(source_data["scene_scale"], dtype=np.float64)
    if values.ndim == 0:
        value = values.reshape(1)
    elif values.ndim == 1:
        if len(values) == total_count:
            value = np.asarray([values[index]])
        elif total_count == 1 and len(values) in {1, 3}:
            value = values
        else:
            raise ValueError(
                f"scene_scale shape {values.shape} does not match {total_count} grasps"
            )
    else:
        if index >= len(values):
            raise ValueError(
                f"scene_scale shape {values.shape} does not contain grasp {index}"
            )
        value = np.asarray(values[index]).reshape(-1)
    if value.size == 1:
        return np.full(3, float(value[0]), dtype=np.float64)
    if value.size == 3:
        return value.astype(np.float64, copy=False)
    raise ValueError(f"Unsupported scene_scale value shape: {value.shape}")


def _filter_scene(scene_path: str, cfg) -> list[dict]:
    from dexonomy.sim import HandCfg, MuJoCo_EvalCfg, MuJoCo_EvalEnv, MuJoCo_OptCfg, MuJoCo_OptEnv

    scene_path = os.path.abspath(scene_path)
    scene_cfg = load_scene_cfg(scene_path)
    target_name = scene_cfg["task"]["obj_name"]
    new_obj_pose = np.asarray(scene_cfg["scene"][target_name]["pose"], dtype=np.float32)
    new_obj_scale = np.asarray(
        scene_cfg["scene"][target_name]["scale"], dtype=np.float64
    )
    if bool(cfg.get("single_object_library", False)):
        # Independent libraries are indexed by model and physical scale, not
        # the source-scene filename originally used to construct clutter.
        source_rel = Path(target_name) / "floating" / "library.npy"
    else:
        source_scene_path = scene_cfg["clutter"]["source_scene_paths"][target_name]
        source_scene_path = os.path.abspath(source_scene_path)
        source_scene_dir = Path(cfg.source_scene_dir).resolve()
        try:
            source_rel = Path(source_scene_path).resolve().relative_to(source_scene_dir)
        except ValueError as exc:
            raise ValueError(
                f"Source scene {source_scene_path} is outside {source_scene_dir}"
            ) from exc

    static_env = MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=str(cfg.hand_xml), freejoint=True),
        scene_cfg=scene_cfg,
        sim_cfg=MuJoCo_OptCfg(obj_margin=0.0, hand_margin=0.0, plane_margin=0.0),
        debug_render=False,
        debug_view=False,
    )
    lift_env = None
    if bool(cfg.lift.enabled):
        lift_scene = _scene_with_mass(scene_cfg, float(cfg.lift.object_mass))
        lift_env = MuJoCo_EvalEnv(
            hand_cfg=HandCfg(xml_path=str(cfg.hand_xml), freejoint=True),
            scene_cfg=lift_scene,
            sim_cfg=MuJoCo_EvalCfg(
                miu_coef=tuple(cfg.lift.friction),
                obj_margin=0.0,
                hand_margin=0.0,
                plane_margin=0.0,
            ),
            debug_render=False,
            debug_view=False,
        )
        lift_env._model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)

    source_grasp_dir = Path(cfg.source_grasp_dir)
    output_dir = Path(cfg.output_dir)
    records = []
    required = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
    for taxonomy in _taxonomy_names(source_grasp_dir, cfg.taxonomies):
        output_path = output_dir / taxonomy / Path(scene_path).parent.name / Path(scene_path).name
        input_path = source_grasp_dir / taxonomy / source_rel
        if bool(cfg.pool_scale_variants) or bool(cfg.get("single_object_library", False)):
            input_paths = sorted(input_path.parent.glob("*.npy"))
        else:
            input_paths = [input_path] if input_path.is_file() else []
        if not input_paths:
            output_path.unlink(missing_ok=True)
            records.append(
                {
                    "scene_path": scene_path,
                    "taxonomy": taxonomy,
                    "status": "missing_source_grasp",
                    "source_grasp_path": str(input_path),
                }
            )
            continue

        candidates = []
        total_count = 0
        rejected_scale = 0
        invalid_paths = []
        for variant_path in input_paths:
            source_data = np.load(variant_path, allow_pickle=True).item()
            if any(key not in source_data for key in required) or "scene_path" not in source_data:
                invalid_paths.append(str(variant_path))
                continue
            variant_scene_path = os.path.abspath(str(source_data["scene_path"]))
            variant_scene = load_scene_cfg(variant_scene_path)
            if target_name not in variant_scene["scene"]:
                invalid_paths.append(str(variant_path))
                continue
            old_obj_cfg = variant_scene["scene"][target_name]
            old_obj_pose = np.asarray(old_obj_cfg["pose"], dtype=np.float32)
            old_obj_scale = np.asarray(old_obj_cfg["scale"], dtype=np.float64)
            transformed = {
                key: _transform_qposes(source_data[key], old_obj_pose, new_obj_pose)
                for key in required
            }
            variant_count = min(len(transformed[key]) for key in required)
            total_count += variant_count
            for index in range(variant_count):
                try:
                    scale_multiplier = _scene_scale_multiplier(
                        source_data, index, variant_count
                    )
                except (IndexError, TypeError, ValueError):
                    rejected_scale += 1
                    continue
                effective_scale = old_obj_scale * scale_multiplier
                relative_scale_error = float(
                    np.max(
                        np.abs(effective_scale - new_obj_scale)
                        / np.maximum(np.abs(new_obj_scale), 1e-12)
                    )
                )
                if (
                    float(cfg.scene_scale_tolerance) >= 0
                    and relative_scale_error > float(cfg.scene_scale_tolerance)
                ):
                    rejected_scale += 1
                    continue
                candidates.append(
                    {
                        "qposes": {key: transformed[key][index] for key in required},
                        "source_path": variant_path,
                        "source_index": index,
                        "scene_scale": scale_multiplier,
                        "effective_object_scale": effective_scale,
                        "relative_scale_error": relative_scale_error,
                    }
                )

        if not candidates and total_count == 0:
            output_path.unlink(missing_ok=True)
            records.append(
                {
                    "scene_path": scene_path,
                    "taxonomy": taxonomy,
                    "status": "invalid_source_grasp" if invalid_paths else "empty",
                    "input_count": 0, "evaluated_count": 0, "valid_count": 0,
                    "source_grasp_paths": [str(path) for path in input_paths],
                    "invalid_source_grasp_paths": invalid_paths,
                }
            )
            continue

        if int(cfg.max_per_target) > 0:
            candidates = candidates[: int(cfg.max_per_target)]
        evaluated_count = len(candidates)

        valid_candidates = []
        rejected_collision = 0
        rejected_target_penetration = 0
        rejected_contact = 0
        rejected_lift = 0
        for candidate in candidates:
            qposes = candidate["qposes"]
            approach = _approach_qposes(
                qposes["pregrasp_qpos"],
                qposes["grasp_qpos"],
                int(cfg.approach_steps),
            )
            if not bool(cfg.get("check_pregrasp", False)):
                # The released floating-object pregrasp may start below the
                # tabletop after the object is placed in clutter.  Ignore only
                # that exact endpoint; keep checking every subsequent approach
                # sample, including the grasp pose.
                approach = approach[1:]
            reasons = [
                _collision_reason(
                    static_env, qpos, cfg, check_target_penetration=True
                )
                for qpos in approach
            ]
            reasons.append(
                _collision_reason(
                    static_env,
                    qposes["squeeze_qpos"],
                    cfg,
                    check_target_penetration=False,
                )
            )
            if "target_penetration" in reasons:
                rejected_target_penetration += 1
                continue
            if any(reason is not None for reason in reasons):
                rejected_collision += 1
                continue
            if bool(cfg.require_target_contact) and not _has_target_contact(
                static_env,
                qposes["squeeze_qpos"],
                float(cfg.target_contact_margin),
            ):
                rejected_contact += 1
                continue
            if lift_env is not None and not _passes_lift(
                lift_env,
                qposes["pregrasp_qpos"],
                qposes["grasp_qpos"],
                qposes["squeeze_qpos"],
                cfg.lift,
            ):
                rejected_lift += 1
                continue
            valid_candidates.append(candidate)

        status = "empty"
        if valid_candidates:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                key: np.stack(
                    [candidate["qposes"][key] for candidate in valid_candidates]
                )
                for key in required
            }
            result["scene_path"] = os.path.relpath(scene_path, os.getcwd())
            result["target_name"] = target_name
            result["source_grasp_paths"] = np.asarray(
                [
                    os.path.relpath(candidate["source_path"], os.getcwd())
                    for candidate in valid_candidates
                ]
            )
            result["source_indices"] = np.asarray(
                [candidate["source_index"] for candidate in valid_candidates],
                dtype=np.int32,
            )
            result["scene_scale"] = np.stack(
                [candidate["scene_scale"] for candidate in valid_candidates]
            )
            result["effective_object_scale"] = np.stack(
                [candidate["effective_object_scale"] for candidate in valid_candidates]
            )
            result["relative_scale_error"] = np.asarray(
                [candidate["relative_scale_error"] for candidate in valid_candidates],
                dtype=np.float32,
            )
            result["clutter_filter"] = {
                "simulator": "mujoco",
                "input_count": total_count,
                "evaluated_count": evaluated_count,
                "valid_count": len(valid_candidates),
                "rejected_scale": rejected_scale,
                "rejected_collision": rejected_collision,
                "rejected_target_penetration": rejected_target_penetration,
                "rejected_target_contact": rejected_contact,
                "rejected_lift": rejected_lift,
            }
            np.save(output_path, result)
            status = "saved"
        elif output_path.exists():
            output_path.unlink()
        records.append(
            {
                "scene_path": scene_path,
                "taxonomy": taxonomy,
                "status": status,
                "source_grasp_paths": [str(path) for path in input_paths],
                "output_path": str(output_path) if valid_candidates else None,
                "input_count": total_count,
                "evaluated_count": evaluated_count,
                "valid_count": len(valid_candidates),
                "rejected_scale": rejected_scale,
                "rejected_collision": rejected_collision,
                "rejected_target_penetration": rejected_target_penetration,
                "rejected_target_contact": rejected_contact,
                "rejected_lift": rejected_lift,
            }
        )
        logging.info(
            "%s %s: %d/%d grasps remain "
            "(evaluated=%d, scale=%d, collision=%d, target_penetration=%d, "
            "target_contact=%d, lift=%d)",
            Path(scene_path).parent.name,
            taxonomy,
            len(valid_candidates),
            total_count,
            evaluated_count,
            rejected_scale,
            rejected_collision,
            rejected_target_penetration,
            rejected_contact,
            rejected_lift,
        )
    return records


def _filter_scene_worker(params) -> tuple[str, list[dict]]:
    scene_path, cfg = params
    return scene_path, _filter_scene(scene_path, cfg)


def _filter_scenes(scene_paths: list[str], cfg, n_worker: int = 1) -> list[dict]:
    paths = sorted(set(scene_paths))
    worker_count = max(1, min(int(n_worker), len(paths)))
    records_by_path: dict[str, list[dict]] = {}
    if worker_count == 1:
        for index, scene_path in enumerate(paths, start=1):
            logging.info(
                "Filter clutter target %d/%d: %s", index, len(paths), scene_path
            )
            records_by_path[scene_path] = _filter_scene(scene_path, cfg)
    else:
        logging.info(
            "Filter %d clutter targets with %d CPU workers",
            len(paths),
            worker_count,
        )
        params = zip(paths, [cfg] * len(paths))
        with multiprocessing.Pool(processes=worker_count) as pool:
            for index, (scene_path, scene_records) in enumerate(
                pool.imap_unordered(_filter_scene_worker, params), start=1
            ):
                records_by_path[scene_path] = scene_records
                logging.info(
                    "Finished clutter target %d/%d: %s",
                    index,
                    len(paths),
                    scene_path,
                )

    records = [
        record
        for scene_path in paths
        for record in records_by_path[scene_path]
    ]
    _log_filter_summary(records)
    return records


def _log_filter_summary(records: list[dict]):
    count_keys = (
        "input_count",
        "evaluated_count",
        "valid_count",
        "rejected_scale",
        "rejected_collision",
        "rejected_target_penetration",
        "rejected_target_contact",
        "rejected_lift",
    )
    totals = {
        key: sum(int(record.get(key, 0) or 0) for record in records)
        for key in count_keys
    }
    logging.info(
        "Clutter grasp filter summary: input=%d, evaluated=%d, valid=%d, "
        "scale=%d, collision=%d, target_penetration=%d, target_contact=%d, lift=%d",
        totals["input_count"],
        totals["evaluated_count"],
        totals["valid_count"],
        totals["rejected_scale"],
        totals["rejected_collision"],
        totals["rejected_target_penetration"],
        totals["rejected_target_contact"],
        totals["rejected_lift"],
    )


def _write_manifest(cfg, stage: str, scene_records: list[dict], filter_records: list[dict]):
    manifest_path = Path(cfg.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as file:
        json.dump(
            {
                "stage": stage,
                "seed": int(cfg.seed),
                "scene_records": _plain(scene_records),
                "filter_records": _plain(filter_records),
            },
            file,
            indent=2,
        )
    logging.info("Wrote clutter manifest to %s", manifest_path)


def operate_clutter(cfg):
    """Hydra entry point for MuJoCo scene generation and grasp filtering."""
    op_cfg = cfg.op
    stage = str(op_cfg.stage)
    if stage not in {"scene", "filter", "all"}:
        raise ValueError("op.stage must be one of: scene, filter, all")
    seed = (
        int(op_cfg.seed)
        if op_cfg.seed is not None
        else int(np.random.SeedSequence().generate_state(1)[0])
    )
    op_cfg.seed = seed
    logging.info("Clutter random seed: %d", seed)
    rng = np.random.default_rng(seed)
    scene_records: list[dict] = []
    filter_records: list[dict] = []

    if stage == "scene":
        scene_paths, scene_records = _generate_scenes(op_cfg.scene, rng, cfg.n_worker)
    elif stage == "filter":
        if not Path(op_cfg.filter.source_grasp_dir).is_dir():
            raise FileNotFoundError(
                f"Single-object grasp library not found: {op_cfg.filter.source_grasp_dir}. "
                "Generate the object library before filtering clutter scenes."
            )
        scene_paths = sorted(glob(str(op_cfg.filter.scene_cfg_glob), recursive=True))
        if not scene_paths:
            raise FileNotFoundError(
                f"No clutter scene configs match {op_cfg.filter.scene_cfg_glob}"
            )
        filter_records = _filter_scenes(scene_paths, op_cfg.filter, cfg.n_worker)
    else:
        scene_paths, scene_records, filter_records = _generate_and_filter_scenes(
            op_cfg, rng, cfg.n_worker
        )
        if not scene_paths:
            logging.warning("No clutter scenes requested; skip grasp filtering")

    _write_manifest(op_cfg, stage, scene_records, filter_records)
    return
