"""Packed storage and CPU workers for exhaustive native Dexonomy generation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

PHASES = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")


def contact_checkpoint_relative(relative, contact_index=None):
    relative = Path(relative).with_suffix("")
    if contact_index is not None:
        if not isinstance(contact_index, (int, np.integer)) or contact_index < 0:
            raise ValueError(f"Invalid hand contact index: {contact_index}")
        relative /= f"contact_{contact_index:03d}"
    return relative


def atomic_npy(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, data)
    temporary.replace(path)


def atomic_archive(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temporary.replace(path)


def retain_valid_attempts(arrays, metadata):
    """Remove failed candidate data, preserving IDs and historical completion counts."""
    count = len(arrays["status"])
    valid = np.asarray(arrays["status"]) == "valid"
    expected_valid = metadata["status_counts"].get("valid", 0)
    if int(valid.sum()) != expected_valid:
        raise ValueError("Archive valid rows do not match its completion metadata")
    if sum(metadata["status_counts"].values()) != metadata["attempted"]:
        raise ValueError("Archive completion counts are inconsistent")
    if count != metadata["attempted"] and not valid.all():
        raise ValueError("Archive contains an incomplete subset of attempts")
    offsets = np.asarray(arrays["pregrasp_trajectory_offsets"])
    trajectory = np.asarray(arrays["pregrasp_trajectory_qpos"])
    if (offsets.shape != (count + 1,) or offsets[0] != 0
            or offsets[-1] != len(trajectory) or np.any(np.diff(offsets) < 0)):
        raise ValueError("Invalid pregrasp trajectory offsets")
    kept = np.flatnonzero(valid)
    result = {}
    for key, value in arrays.items():
        if key in ("pregrasp_trajectory_offsets", "pregrasp_trajectory_qpos"):
            continue
        if np.ndim(value) < 1 or len(value) != count:
            raise ValueError(f"Unexpected candidate array dimensions: {key}")
        result[key] = value[valid].copy()
    segments = [trajectory[offsets[index]:offsets[index + 1]] for index in kept]
    result["pregrasp_trajectory_qpos"] = np.concatenate(segments) if segments else trajectory[:0].copy()
    lengths = offsets[kept + 1] - offsets[kept]
    result["pregrasp_trajectory_offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    metadata = dict(metadata, archive_scope="valid_only", stored_candidates=int(valid.sum()),
                    discarded_failed_candidates=metadata["attempted"] - int(valid.sum()))
    return result, metadata


def store_native_sampling(output, scene_path, epoch, fingerprint, points, inplane,
                          rotations, translations, obj_scale, obj_pose, source_relative=None):
    """Checkpoint unmodified surface seeds so all types and resumes share them."""
    from transforms3d.quaternions import quat2mat

    scene_path = Path(scene_path)
    relative = (Path(source_relative).with_suffix("") if source_relative is not None
                else Path(scene_path.parent.name) / scene_path.stem)
    path = Path(output) / "sampling" / relative / f"epoch_{epoch:03d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["run_fingerprint"] != fingerprint:
                raise ValueError(f"Different generation settings in {path}")
            return archive["rot_o2c_init"].copy(), archive["trans_o2c_init"].copy(), path

    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    if rotations.shape != (points * inplane, 3, 3) or translations.shape != (points * inplane, 3):
        raise ValueError("Unexpected native surface sample dimensions")
    if not np.allclose(translations.reshape(inplane, points, 3),
                       np.broadcast_to(translations[:points], (inplane, points, 3))):
        raise ValueError("Native orientations do not share the same surface points")
    world_rotation = quat2mat(obj_pose[3:])
    frames_world = np.einsum("ij,njk->nik", world_rotation, rotations)
    points_world = (translations[:points] * obj_scale) @ world_rotation.T + obj_pose[:3]
    metadata = {
        "format": "dexonomy_native_surface_samples_v1", "run_fingerprint": fingerprint,
        "scene_path": str(scene_path), "target": scene_path.stem, "epoch": epoch,
        "point_count": points, "inplane_count": inplane, "source": "native_generation",
        "sampler": "dexonomy.data.obj_loader.sample_init_pose", "original_sampling_saved": True,
    }
    atomic_archive(path, {
        "rot_o2c_init": rotations, "trans_o2c_init": translations,
        "points_object": translations[:points], "points_world": points_world,
        "inward_normals_world": frames_world[:points, :, 0], "frames_world": frames_world,
        "point_indices": np.tile(np.arange(points), inplane),
        "roll_indices": np.repeat(np.arange(inplane), points),
        "roll_angles_rad": np.linspace(-np.pi, np.pi, inplane),
    }, metadata)
    return rotations, translations, path


def check_clutter_grasp(env, qposes, cfg):
    from dexonomy.op.gen_clutter import _approach_qposes, _collision_reason, _has_target_contact

    approach = _approach_qposes(qposes[0], qposes[1], int(cfg.approach_steps))
    if not cfg.check_pregrasp:
        approach = approach[1:]
    for pose in approach:
        reason = _collision_reason(env, pose, cfg, check_target_penetration=True)
        if reason:
            return "clutter_" + reason
    reason = _collision_reason(env, qposes[2], cfg, check_target_penetration=False)
    if reason:
        return "clutter_" + reason
    if cfg.require_target_contact and not _has_target_contact(env, qposes[2], float(cfg.target_contact_margin)):
        return "clutter_target_contact"
    return None


def process_native_batch(job):
    """Optimize every seed with the original native optimizer and evaluator."""
    from omegaconf import OmegaConf
    from dexonomy.op.gen_grasp import optimize_grasp_data
    from dexonomy.op.eval_data import evaluate_grasp_data
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    output = Path(job["output"])
    scene_path = Path(job["scene_path"])
    taxonomy, epoch = job["taxonomy"], job["epoch"]
    scene = load_scene_cfg(str(scene_path))
    target = scene["task"]["obj_name"]
    single_object = job.get("single_object", False)
    if single_object and (set(scene["scene"]) != {target} or
                          scene["scene"][target]["type"] != "rigid_object"):
        raise ValueError("Single-object generation requires one rigid object and no scene obstacles")
    relative = (Path(job["source_relative"]).with_suffix("") if single_object
                else Path(scene_path.parent.name) / target)
    contact_index = job.get("hand_contact_index")
    contact_count = job.get("hand_contact_count", 1)
    if contact_index is not None:
        if (not single_object or not 0 <= contact_index < contact_count or
                contact_count != len(job["hand_cbody"])):
            raise ValueError("Invalid exhaustive hand contact job")
        relative = contact_checkpoint_relative(relative, contact_index)
    archive_path = output / "attempts" / taxonomy / relative / f"epoch_{epoch:03d}.npz"
    # Atomic archives are also checkpoints if the parent died before updating SQLite.
    if archive_path.exists():
        with np.load(archive_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        if metadata["run_fingerprint"] != job["run_fingerprint"]:
            raise ValueError(f"Different generation settings in {archive_path}")
        return metadata

    grasp_cfg = OmegaConf.create(job["grasp_config"])
    eval_cfg = OmegaConf.create(job["eval_config"])
    filter_cfg = OmegaConf.create(job["filter_config"])
    hand_cfg = HandCfg(xml_path=grasp_cfg.hand.xml_path, freejoint=True)
    opt_env = MuJoCo_OptEnv(
        hand_cfg=hand_cfg, scene_cfg=scene,
        sim_cfg=MuJoCo_OptCfg(obj_margin=grasp_cfg.op.pregrasp.cdist),
        debug_render=False, debug_view=False,
    )
    eval_env, collision_env = None, None
    inputs = job["inputs"]
    count = len(inputs["grasp_qpos"])
    initial_qpos = np.asarray(inputs["grasp_qpos"], dtype=np.float32)
    arrays = {phase: np.full((count, 29), np.nan, dtype=np.float32) for phase in PHASES}
    arrays["grasp_qpos"][:] = initial_qpos
    statuses = np.full(count, "unprocessed", dtype="U64")
    errors = np.full(count, "", dtype="U256")
    trajectories, offsets = [], [0]
    for index in range(count):
        data = {
            "hand_name": "shadow", "tmpl_name": taxonomy,
            "hand_cbody": list(job["hand_cbody"]),
            "required_cbody": job["required_cbody"],
            "grasp_qpos": initial_qpos[index:index + 1].copy(),
            "obj_cpn_w": inputs["obj_cpn_w"][index].copy(),
            "hand_cpn_w": inputs["hand_cpn_w"][index].copy(),
            "ext_center": inputs["ext_center"][index].copy(),
            "ext_wrench": inputs["ext_wrench"][index].copy(),
            "n_evo": inputs["n_evo"][index].copy(),
            "obj_pose": inputs["obj_pose"][index].copy(),
            "obj_scale": inputs["obj_scale"][index].copy(),
            "scene_path": str(scene_path),
        }
        try:
            if not np.isfinite(initial_qpos[index]).all():
                statuses[index] = "init_numerical_failure"
                continue
            data, optimized, reason = optimize_grasp_data(data, grasp_cfg, opt_env)
            statuses[index] = reason
            if optimized:
                passed, eval_env = evaluate_grasp_data(data, eval_cfg, scene, eval_env)
                statuses[index] = "eval_failed"
                if passed:
                    if single_object:
                        statuses[index] = "valid"
                    elif collision_env is None:
                        collision_env = MuJoCo_OptEnv(
                            hand_cfg=hand_cfg, scene_cfg=scene,
                            sim_cfg=MuJoCo_OptCfg(obj_margin=0.0, hand_margin=0.0, plane_margin=0.0),
                            debug_render=False, debug_view=False,
                        )
                    if not single_object:
                        qposes = [np.asarray(data[phase])[0] for phase in PHASES]
                        statuses[index] = check_clutter_grasp(collision_env, qposes, filter_cfg) or "valid"
        except Exception as exc:
            statuses[index] = "error"
            errors[index] = f"{type(exc).__name__}: {exc}"
            # Discard mutable simulation state after unexpected exceptions.
            eval_env = None
            opt_env = MuJoCo_OptEnv(
                hand_cfg=hand_cfg, scene_cfg=scene,
                sim_cfg=MuJoCo_OptCfg(obj_margin=grasp_cfg.op.pregrasp.cdist),
                debug_render=False, debug_view=False,
            )
        finally:
            for phase in PHASES:
                if phase in data:
                    arrays[phase][index] = np.asarray(data[phase])[0]
            trajectory = np.asarray(data.get("pregrasp_qpos", np.empty((0, 29))), dtype=np.float32)
            trajectories.append(trajectory)
            offsets.append(offsets[-1] + len(trajectory))

    indices = np.arange(count, dtype=np.int32)
    candidate_offset = ((epoch * contact_count + contact_index) * count
                        if contact_index is not None else epoch * count)
    arrays.update(
        candidate_indices=candidate_offset + indices,
        sample_point_indices=indices % job["points"],
        inplane_indices=indices // job["points"],
        init_grasp_qpos=initial_qpos, status=statuses, error_messages=errors,
        init_obj_cpn_w=inputs["obj_cpn_w"], init_hand_cpn_w=inputs["hand_cpn_w"],
        ext_center=inputs["ext_center"], ext_wrench=inputs["ext_wrench"], n_evo=inputs["n_evo"],
        pregrasp_trajectory_qpos=np.concatenate(trajectories),
        pregrasp_trajectory_offsets=np.asarray(offsets, dtype=np.int64),
    )
    if contact_index is not None:
        if count != job["points"] * job["inplane"]:
            raise ValueError("Exhaustive contact batch must cover every object point/roll")
        arrays["hand_contact_indices"] = np.full(count, contact_index, dtype=np.int32)
    status_counts = dict(Counter(statuses.tolist()))
    metadata = {
        "format": "dexonomy_native_attempts_v1", "run_fingerprint": job["run_fingerprint"],
        "scene": scene_path.parent.name, "target": target, "taxonomy": taxonomy,
        "scene_path": str(scene_path), "epoch": epoch, "points": job["points"],
        "inplane": job["inplane"], "attempted": count,
        "status_counts": status_counts, "archive_path": str(archive_path),
        "hand_cbody": job["hand_cbody"], "required_cbody": job["required_cbody"],
        "obj_pose": inputs["obj_pose"][0].tolist(), "obj_scale": inputs["obj_scale"][0].tolist(),
        "sampling_archive": job.get("sampling_archive", ""),
    }
    if single_object:
        metadata.update(single_object=True, source_relative=job["source_relative"],
                        job_key=job["job_key"], format="dexonomy_single_object_attempts_v1")
        if "library_root" in job:
            metadata["library_root"] = job["library_root"]
    if contact_index is not None:
        metadata.update(hand_contact_index=contact_index, hand_contact_count=contact_count,
                        hand_contact_body=job["hand_cbody"][contact_index], hand_contact_mode="all")
    valid = statuses == "valid"
    validated_path = output / "validated" / taxonomy / relative / f"epoch_{epoch:03d}.npy"
    atomic_npy(validated_path, {
        **{phase: arrays[phase][valid] for phase in PHASES},
        "scene_path": str(scene_path), "target_name": target,
        "source_grasp_paths": np.full(int(valid.sum()), str(archive_path)),
        "source_indices": arrays["candidate_indices"][valid],
        **({"hand_contact_indices": arrays["hand_contact_indices"][valid]}
           if contact_index is not None else {}),
    })
    if not job.get("keep_failed", True):
        arrays, metadata = retain_valid_attempts(arrays, metadata)
    # Commit archive last, so a completed archive always has its validated output.
    atomic_archive(archive_path, arrays, metadata)
    return metadata


def merge_validated_target(metadata, output, base_root):
    """Keep every validated row; no cap, FPS or deduplication is applied."""
    if metadata.get("single_object"):
        return merge_single_object_target(metadata, output)
    output, base_root = Path(output), Path(base_root)
    taxonomy, scene_name, target = metadata["taxonomy"], metadata["scene"], metadata["target"]
    sources = []
    base_path = base_root / taxonomy / scene_name / f"{target}.npy"
    if base_path.exists():
        sources.append((base_path, "clutter_filter"))
    sources.extend((path, "dexonomy_eval+clutter_filter") for path in sorted(
        (output / "validated" / taxonomy / scene_name / target).glob("epoch_*.npy")))
    poses = {phase: [] for phase in PHASES}
    paths, indices, verifiers = [], [], []
    existing = 0
    for path, verifier in sources:
        data = np.load(path, allow_pickle=True).item()
        count = len(data["grasp_qpos"])
        if any(len(data[phase]) != count for phase in PHASES):
            raise ValueError(f"Mismatched phase counts: {path}")
        for phase in PHASES:
            poses[phase].append(np.asarray(data[phase], dtype=np.float32))
        if verifier == "clutter_filter":
            existing += count
            paths.extend([str(path)] * count)
            indices.extend(range(count))
        else:
            paths.extend(data["source_grasp_paths"].tolist())
            indices.extend(data["source_indices"].tolist())
        verifiers.extend([verifier] * count)
    destination = output / "clutter/filtered_grasps" / taxonomy / scene_name / f"{target}.npy"
    count = len(paths)
    if count:
        atomic_npy(destination, {
            **{phase: np.concatenate(values) for phase, values in poses.items()},
            "scene_path": metadata["scene_path"], "target_name": target,
            "source_grasp_paths": np.asarray(paths), "source_indices": np.asarray(indices, dtype=np.int32),
            "verified_by": np.asarray(verifiers), "scene_scale": np.ones((count, 3), dtype=np.float32),
            "effective_object_scale": np.tile(metadata["obj_scale"], (count, 1)),
        })
    else:
        destination.unlink(missing_ok=True)
    return existing


def merge_single_object_target(metadata, output):
    """Publish a reusable object/scale library, independent of clutter scene IDs."""
    output = Path(output)
    library_root = Path(metadata.get("library_root", output))
    relative = Path(metadata["source_relative"])
    runs = dict.fromkeys([library_root, *sorted((library_root / "runs").glob("*")), output])
    samples = []
    for run in runs:
        directory = run / "validated" / metadata["taxonomy"] / relative.with_suffix("")
        checkpoints = [*directory.glob("epoch_*.npy"), *directory.glob("contact_*/epoch_*.npy")]
        for path in sorted(checkpoints):
            archive = run / "attempts" / path.relative_to(run / "validated").with_suffix(".npz")
            if archive.is_file():
                samples.append(np.load(path, allow_pickle=True).item())
    if not samples:
        raise ValueError(f"No committed validation checkpoints for {relative}")
    count = sum(len(sample["grasp_qpos"]) for sample in samples)
    atomic_npy(library_root / "succ_collect" / metadata["taxonomy"] / relative, {
        **{phase: np.concatenate([sample[phase] for sample in samples]) for phase in PHASES},
        "scene_path": metadata["scene_path"], "target_name": metadata["target"],
        "source_grasp_paths": np.concatenate([sample["source_grasp_paths"] for sample in samples]),
        "source_indices": np.concatenate([sample["source_indices"] for sample in samples]),
        "hand_contact_indices": np.concatenate([
            sample.get("hand_contact_indices", np.full(len(sample["grasp_qpos"]), -1, dtype=np.int32))
            for sample in samples]),
        "scene_scale": np.ones((count, 3), dtype=np.float32),
        "effective_object_scale": np.tile(metadata["obj_scale"], (count, 1)),
        "verified_by": np.full(count, "single_object_native_eval"),
    })
    return 0
