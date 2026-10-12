#!/usr/bin/env python3
"""Run Dexonomy's native init -> grasp -> eval for existing clutter targets."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import sqlite3

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", default="output/clutter_dgn_shadow/clutter/scene_cfg")
    parser.add_argument("--scene", default="scene_*", help="Scene name or glob; scene_* selects all scenes.")
    parser.add_argument("--target", default="*", help="Target object name or glob.")
    parser.add_argument("--output", default="output/clutter_dgn_native_all_shadow")
    parser.add_argument("--taxonomies", nargs="+", help="Default: all 33 native taxonomy templates.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--inplane", type=int, default=8)
    parser.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=True,
                        help="Optimize every points*inplane candidate without initialization selection.")
    parser.add_argument("--keep-failed", action=argparse.BooleanOptionalAction, default=True,
                        help="Archive failed candidate data too; --no-keep-failed saves only valid rows.")
    parser.add_argument("--candidates", type=int, help="Legacy mode only: initialization cap with --no-save-all.")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=84, help="CPU processes for exhaustive native generation.")
    parser.add_argument("--collect-only", action="store_true", help="Collect existing native successes without generating.")
    parser.add_argument("--dry-run", action="store_true", help="Print native commands without writing or running them.")
    args = parser.parse_args()
    for name in ("epochs", "points", "inplane", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.save_all and args.candidates is not None:
        parser.error("--save-all retains points*inplane candidates; remove --candidates")
    if not args.save_all and not args.keep_failed:
        parser.error("--no-keep-failed applies to exhaustive --save-all mode")
    if args.candidates is not None and args.candidates < 1:
        parser.error("--candidates must be positive")
    # Legacy serial grasp/eval do not write their processed-path logs.
    if not args.save_all and args.workers < 2:
        parser.error("--workers must be at least 2 for resumable native grasp/eval")
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    return args


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Native generation is already running in {output}") from None
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def select_taxonomies(args, base_root: Path) -> list[str]:
    templates = REPO_ROOT / "assets/hand/shadow/init_tmpl"
    if args.taxonomies:
        names = args.taxonomies
    elif not args.save_all and base_root.is_dir():
        names = [path.name for path in base_root.iterdir() if path.is_dir()]
    else:
        names = [path.stem for path in templates.glob("[0-9]*.npy")]
    for name in names:
        if not (templates / f"{name}.npy").is_file():
            raise ValueError(f"Native template does not exist: {name}")
    if not names:
        raise ValueError("No native templates selected")
    return sorted(set(names), key=lambda name: (int(name.split("_", 1)[0]), name))


def template_contact_counts(taxonomies):
    counts = {}
    for name in taxonomies:
        template = np.load(REPO_ROOT / f"assets/hand/shadow/init_tmpl/{name}.npy", allow_pickle=True).item()
        points = np.asarray(template["hand_cpn_w"])
        if points.ndim != 2 or points.shape[1] != 6 or not len(points):
            raise ValueError(f"Invalid hand template contact points: {name}")
        if len(points) != len(template["hand_cbody"]):
            raise ValueError(f"Hand template contact bodies do not match its points: {name}")
        counts[name] = len(points)
    return counts


def native_command(args, output: Path, cfg_glob: str, taxonomy: str, stage: str) -> list[str]:
    command = [
        sys.executable, "-m", "dexonomy.main", f"op={stage}",
        f"save_dir={json.dumps(str(output))}", "exp_name=clutter_native",
        f"tmpl_name={taxonomy}", "skip_done=true", "skip_traj=true",
        "tmpl_upd_mode=disabled", f"log_id={taxonomy}",
        # Select only this taxonomy, including when resuming a partially finished run.
        f"debug_name={json.dumps('/' + taxonomy + '/')}",
        f"n_worker={1 if stage == 'init' else args.workers}",
    ]
    if stage == "init":
        command.extend([
            f"init_gpu=[{args.gpu}]", f"op.epoch={args.epochs}",
            f"op.object.cfg_path={json.dumps(cfg_glob)}", "op.object.n_cfg=0",
            # A one-target batch keeps native skip_done from skipping unfinished
            # targets when another target in the same batch already has outputs.
            "op.object.batch_size=1", f"op.object.n_init_point={args.points}",
            f"op.object.n_init_inplane={args.inplane}",
            f"op.filter.general.n_final={args.candidates or 20}",
        ])
    return command


def run_native(command: list[str], output: Path, taxonomy: str, stage: str) -> None:
    log_path = output / "runner_logs" / f"{taxonomy}_{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{taxonomy}: {stage}; log: {log_path}", flush=True)
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    with log_path.open("w") as log:
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    # Some native operations catch exceptions and return exit code zero.
    with log_path.open() as log:
        if any("Traceback (most recent call last)" in line for line in log):
            raise RuntimeError(f"Native {stage} raised an exception; see {log_path}")
    print(f"{taxonomy}: {stage} finished", flush=True)


def collect_results(scene_paths: list[Path], base_root: Path, output: Path) -> dict:
    """Pack native successes for the current viewer, preserving existing grasps."""
    from omegaconf import OmegaConf
    from dexonomy.op.gen_clutter import _approach_qposes, _collision_reason, _has_target_contact
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    from dexonomy.util.file_util import load_scene_cfg

    filter_cfg = OmegaConf.load(REPO_ROOT / "dexonomy/config/op/clutter.yaml").filter
    packed_root = output / "clutter/filtered_grasps"
    totals = Counter()
    per_target = []
    for scene_path in scene_paths:
        scene = load_scene_cfg(str(scene_path))
        target = scene["task"]["obj_name"]
        scene_name = scene_path.parent.name
        native_dirs = {path.parent.parent.parent.name: path
                       for path in (output / "succ_grasp").glob(f"*/clutter/{scene_name}/{target}")}
        base_files = {path.parent.parent.name: path
                      for path in base_root.glob(f"*/{scene_name}/{scene_path.name}")}
        env = None
        counts = Counter()
        for taxonomy in sorted(set(native_dirs) | set(base_files)):
            rows, provenance, seen = [], [], set()

            def add_row(qposes, source, source_index, verifier):
                values = tuple(np.asarray(qpos, dtype=np.float32) for qpos in qposes)
                if any(value.shape != (29,) or not np.isfinite(value).all() for value in values):
                    raise ValueError(f"Invalid Shadow hand pose in {source}")
                key = b"".join(value.tobytes() for value in values)
                if key in seen:
                    counts["duplicates"] += 1
                    return
                seen.add(key)
                rows.append(values)
                provenance.append((str(source), source_index, verifier))
                counts["existing" if verifier == "clutter_filter" else "new"] += 1

            if taxonomy in base_files:
                path = base_files[taxonomy]
                data = np.load(path, allow_pickle=True).item()
                arrays = [np.asarray(data[phase]).reshape(-1, 29) for phase in PHASES]
                if len({len(array) for array in arrays}) != 1:
                    raise ValueError(f"Mismatched pose counts in {path}")
                for index, qposes in enumerate(zip(*arrays)):
                    add_row(qposes, path, index, "clutter_filter")
            if taxonomy in native_dirs:
                for path in sorted(native_dirs[taxonomy].glob("*.npy")):
                    data = np.load(path, allow_pickle=True).item()
                    if Path(str(data["scene_path"])).resolve() != scene_path:
                        raise ValueError(f"Native sample uses a different scene config: {path}")
                    qposes = [np.asarray(data[phase]).reshape(-1, 29)[0] for phase in PHASES]
                    counts["native_successes"] += 1
                    if env is None:
                        env = MuJoCo_OptEnv(
                            hand_cfg=HandCfg(xml_path=str(REPO_ROOT / "assets/hand/shadow/right.xml"), freejoint=True),
                            scene_cfg=scene,
                            sim_cfg=MuJoCo_OptCfg(obj_margin=0.0, hand_margin=0.0, plane_margin=0.0),
                            debug_render=False, debug_view=False,
                        )
                    approach = _approach_qposes(qposes[0], qposes[1], int(filter_cfg.approach_steps))
                    if not filter_cfg.check_pregrasp:
                        approach = approach[1:]
                    reasons = [_collision_reason(env, pose, filter_cfg, check_target_penetration=True)
                               for pose in approach]
                    reasons.append(_collision_reason(env, qposes[2], filter_cfg, check_target_penetration=False))
                    reason = next((reason for reason in reasons if reason), None)
                    if reason:
                        counts[f"rejected_{reason}"] += 1
                        continue
                    if filter_cfg.require_target_contact and not _has_target_contact(env, qposes[2], float(filter_cfg.target_contact_margin)):
                        counts["rejected_target_contact"] += 1
                        continue
                    add_row(qposes, path, 0, "dexonomy_eval+clutter_filter")
            destination = packed_root / taxonomy / scene_name / scene_path.name
            if rows:
                result = {phase: np.stack([row[index] for row in rows]) for index, phase in enumerate(PHASES)}
                result.update(
                    scene_path=str(scene_path), target_name=target,
                    source_grasp_paths=np.asarray([item[0] for item in provenance]),
                    source_indices=np.asarray([item[1] for item in provenance], dtype=np.int32),
                    verified_by=np.asarray([item[2] for item in provenance]),
                    scene_scale=np.ones((len(rows), 3), dtype=np.float32),
                    effective_object_scale=np.tile(scene["scene"][target]["scale"], (len(rows), 1)),
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".npy.tmp")
                with temporary.open("wb") as file:
                    np.save(file, result)
                temporary.replace(destination)
            else:
                destination.unlink(missing_ok=True)
        per_target.append({"scene": scene_name, "target": target, **dict(counts)})
        totals.update(counts)
        if counts["new"]:
            print(f"{scene_name}/{target}: {counts['existing']} existing + {counts['new']} new", flush=True)
    summary = {"source_scene_count": len({path.parent for path in scene_paths}),
               "target_count": len(scene_paths), "totals": dict(totals), "targets": per_target,
               "filtered_grasp_root": str(packed_root)}
    summary_path = output / "clutter/native_generation_statistics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Collected {dict(totals)}; viewer root: {packed_root}", flush=True)
    return summary


def native_configs(args, output, cfg_glob):
    from omegaconf import OmegaConf

    base = OmegaConf.load(REPO_ROOT / "dexonomy/config/base.yaml")
    configs = {}
    for stage in ("init", "grasp", "eval"):
        cfg = OmegaConf.merge(base, {
            "op": OmegaConf.load(REPO_ROOT / f"dexonomy/config/op/{stage}.yaml"),
            "hand": OmegaConf.load(REPO_ROOT / "dexonomy/config/hand/shadow.yaml"),
            "op_name": stage, "hand_name": "shadow", "asset_root": str(REPO_ROOT / "assets"),
            "save_dir": str(output), "n_worker": args.workers, "skip_traj": True,
            "tmpl_upd_mode": "disabled", "debug_view": False, "debug_render": False,
        })
        if stage == "init":
            cfg.op.object.update(cfg_path=cfg_glob, n_cfg=0, batch_size=1,
                                 n_init_point=args.points, n_init_inplane=args.inplane)
            cfg.op.epoch = args.epochs
        configs[stage] = OmegaConf.to_container(cfg, resolve=True)
    configs["filter"] = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "dexonomy/config/op/clutter.yaml").filter)
    # Worker only uses geometric checks; remove unresolved path interpolations.
    for field in ("scene_cfg_glob", "source_scene_dir", "output_dir", "hand_xml", "single_object_library"):
        configs["filter"].pop(field, None)
    return configs


def initialize_selected_progress(database, is_selected):
    """Retain all library checkpoints; count only this run's selected jobs."""
    database.execute("CREATE TEMP TABLE selected_completed (job_key TEXT PRIMARY KEY)")
    database.execute("CREATE TEMP TABLE selected_status_totals (status TEXT PRIMARY KEY, count INTEGER NOT NULL)")
    completed, statuses = set(), Counter()
    for key, raw in database.execute("SELECT job_key, metadata FROM completed"):
        if is_selected(key):
            completed.add(key)
            statuses.update(json.loads(raw)["status_counts"])
    database.executemany("INSERT INTO selected_completed VALUES (?)", ((key,) for key in completed))
    database.executemany("INSERT INTO selected_status_totals VALUES (?, ?)", statuses.items())
    database.commit()
    return completed


def write_progress_summary(database, settings, output, *, library_root=None, history=(0, 0)):
    library_root = Path(library_root) if library_root is not None else output
    prefix = "selected_" if settings.get("single_object") else ""
    statuses = dict(database.execute(f"SELECT status, count FROM {prefix}status_totals"))
    completed = database.execute(f"SELECT COUNT(*) FROM {prefix}completed").fetchone()[0]
    summary = {
        "source_scene_count": settings["scene_count"], "target_count": settings["target_count"],
        "taxonomy_count": len(settings["taxonomies"]), "points": settings["points"],
        "inplane": settings["inplane"], "epochs": settings["epochs"],
        "planned_groups": settings.get("planned_groups", settings["target_count"] * len(settings["taxonomies"]) * settings["epochs"]),
        "planned_attempts": settings["planned_attempts"], "completed_groups": completed,
        "attempted": sum(statuses.values()), "new_valid_grasps": statuses.get("valid", 0),
        "status_counts": statuses,
        "filtered_grasp_root": str(output / "clutter/filtered_grasps"),
        "attempt_archive_root": str(output / "attempts"),
    }
    path = output / "clutter/native_generation_statistics.json"
    if settings.get("single_object"):
        summary.update(stage="single_object", object_count=settings["scene_count"],
                       object_scale_count=settings["target_count"],
                       library_completed_groups=history[0] + database.execute("SELECT COUNT(*) FROM completed").fetchone()[0],
                       library_valid_grasps=history[1] + dict(database.execute("SELECT status, count FROM status_totals")).get("valid", 0),
                       successful_grasp_root=str(library_root / "succ_collect"),
                       run_directory=str(output), run_config=str(output / "run_config.json"))
        summary.pop("filtered_grasp_root")
        if settings.get("hand_contact_mode") == "all":
            summary.update(hand_contact_mode="all", hand_contact_counts=settings["hand_contact_counts"],
                           hand_contact_count_total=sum(settings["hand_contact_counts"].values()),
                           candidates_per_hand_contact=settings["points"] * settings["inplane"])
        path = output / "progress.json"
    paths = {path}
    if settings.get("single_object"):
        paths.add(library_root / "progress.json")
    for destination in paths:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(destination)
    return summary


def run_all_candidates(args, scene_paths, base_root, output, taxonomies, cfg_glob):
    """Use the native matcher and native CPU pipeline with bounded packed jobs."""
    from omegaconf import OmegaConf
    from dexonomy.op.native_clutter_batch import (
        process_native_batch, merge_validated_target, store_native_sampling, contact_checkpoint_relative,
    )

    configs = native_configs(args, output, cfg_glob)
    single_object = getattr(args, "single_object", False)
    all_contacts = getattr(args, "all_hand_contacts", False)
    if all_contacts and not single_object:
        raise ValueError("Exhaustive hand contact anchors require the isolated object pipeline")
    contact_counts = template_contact_counts(taxonomies) if all_contacts else {name: 1 for name in taxonomies}
    library_root = Path(getattr(args, "library_root", output)) if single_object else output
    scene_root = resolve_path(args.scene_root)

    def source_relative(path):
        return (path.relative_to(scene_root) if single_object
                else Path(path.parent.name) / path.name)

    def contact_indices(taxonomy):
        return range(contact_counts[taxonomy]) if all_contacts else (None,)

    def job_key(path, taxonomy, epoch, contact_index=None):
        return f"{taxonomy}/{contact_checkpoint_relative(source_relative(path), contact_index)}/{epoch}"

    def attempt_path(path, taxonomy, epoch, contact_index=None):
        return (output / "attempts" / taxonomy /
                contact_checkpoint_relative(source_relative(path), contact_index) / f"epoch_{epoch:03d}.npz")

    selected_targets = {str(source_relative(path).with_suffix("")) for path in scene_paths}
    selected_paths = {str(path.resolve()) for path in scene_paths}
    selected_types = set(taxonomies)

    def is_selected(key):
        taxonomy, remainder = key.split("/", 1)
        relative, epoch = remainder.rsplit("/", 1)
        if all_contacts:
            relative, anchor = relative.rsplit("/", 1)
            if (taxonomy not in contact_counts or
                    anchor not in {f"contact_{index:03d}" for index in contact_indices(taxonomy)}):
                return False
        return (taxonomy in selected_types and relative in selected_targets and
                0 <= int(epoch) < args.epochs)

    def generation_policy(value):
        # Extending an object library must reuse previously completed object/scale jobs.
        return {key: item for key, item in value.items()
                if key not in {"scene_count", "target_count", "planned_attempts", "planned_groups", "source_sha256"}}
    print(f"Attempt archive retention: {'all candidates' if args.keep_failed else 'valid candidates only'}; "
          "all candidates are still optimized and counted.", flush=True)
    settings = {
        "format": "dexonomy_native_all_v1", "scene_root": str(resolve_path(args.scene_root)),
        "scene": args.scene, "target": args.target,
        "scene_count": len({path.parent for path in scene_paths}), "target_count": len(scene_paths),
        "taxonomies": taxonomies, "epochs": args.epochs, "points": args.points, "inplane": args.inplane,
        "planned_attempts": len(scene_paths) * len(taxonomies) * args.epochs * args.points * args.inplane,
        "native_matcher": configs["init"]["op"]["matcher"],
        "native_grasp": configs["grasp"]["op"], "native_eval": configs["eval"]["op"],
        "clutter_filter": configs["filter"], "object_mass": configs["init"]["obj_mass"],
        "hand_model_sha256": hashlib.sha256((REPO_ROOT / "assets/hand/shadow/right.xml").read_bytes()).hexdigest(),
        "template_sha256": {name: hashlib.sha256((REPO_ROOT / f"assets/hand/shadow/init_tmpl/{name}.npy").read_bytes()).hexdigest()
                            for name in taxonomies},
    }
    # Detect changed source configs on resume, even when target filenames stayed the same.
    source_hash = hashlib.sha256()
    for path in scene_paths:
        source_hash.update(str(path).encode())
        source_hash.update(path.read_bytes())
    settings["source_sha256"] = source_hash.hexdigest()
    if single_object:
        settings.update(format="dexonomy_single_object_v1", single_object=True,
                        scene_count=len({path.parts[-3] for path in scene_paths}))
        settings.pop("clutter_filter")
    if all_contacts:
        groups = len(scene_paths) * sum(contact_counts.values()) * args.epochs
        settings.update(hand_contact_mode="all", hand_contact_counts=contact_counts,
                        planned_groups=groups, planned_attempts=groups * args.points * args.inplane)
    fingerprint = hashlib.sha256(json.dumps(
        generation_policy(settings) if single_object else settings, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        print("Every matched candidate will run native grasp optimization; no init selection or FPS cap.")
        return

    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "run_config.json"
    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if (generation_policy(saved) != generation_policy(settings) if single_object else saved != settings):
            raise ValueError(f"Generation settings or source configs changed; use a new --output. See {settings_path}")
        if single_object:
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    # Make the updated dataset immediately usable while native generation runs.
    # Never replace already merged results during a restart.
    baseline_marker = output / "baseline_copied.json"
    if not single_object and not baseline_marker.exists():
        selected = {(path.parent.name, path.name) for path in scene_paths}
        copied = 0
        for source in base_root.glob("*/*/*.npy"):
            if (source.parent.name, source.name) not in selected:
                continue
            destination = output / "clutter/filtered_grasps" / source.relative_to(base_root)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied += 1
        baseline_marker.write_text(json.dumps({"copied_files": copied}) + "\n")
        print(f"Copied {copied} existing grasp files into the updated dataset", flush=True)
    database = sqlite3.connect(output / "progress.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE IF NOT EXISTS completed (job_key TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
    database.execute("CREATE TABLE IF NOT EXISTS status_totals (status TEXT PRIMARY KEY, count INTEGER NOT NULL)")
    completed = (initialize_selected_progress(database, is_selected) if single_object
                 else {row[0] for row in database.execute("SELECT job_key FROM completed")})
    history = [0, 0]
    if single_object:
        for directory in [library_root, *sorted((library_root / "runs").glob("*"))]:
            checkpoint = directory / "progress.sqlite"
            if directory == output or not checkpoint.is_file():
                continue
            previous = sqlite3.connect(f"file:{checkpoint}?mode=ro", uri=True)
            try:
                history[0] += previous.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
                history[1] += dict(previous.execute("SELECT status, count FROM status_totals")).get("valid", 0)
            finally:
                previous.close()

    def summarize():
        return write_progress_summary(database, settings, output, library_root=library_root, history=history)

    initial = summarize()
    print(f"Resuming {initial['completed_groups']}/{initial['planned_groups']} groups; "
          f"{initial['attempted']} attempts completed; {initial['new_valid_grasps']} new valid grasps", flush=True)

    def record(metadata):
        key = metadata.get("job_key", f"{metadata['taxonomy']}/{metadata['scene']}/{metadata['target']}/{metadata['epoch']}")
        if single_object and not is_selected(key):
            return
        if single_object:
            metadata = dict(metadata, library_root=str(library_root))
        merge_validated_target(metadata, output, base_root)
        if key not in completed:
            with database:
                database.execute("INSERT INTO completed VALUES (?, ?)", (key, json.dumps(metadata)))
                for status, count in metadata["status_counts"].items():
                    database.execute("INSERT INTO status_totals VALUES (?, ?) ON CONFLICT(status) "
                                     "DO UPDATE SET count=count+excluded.count", (status, count))
                if single_object:
                    database.execute("INSERT INTO selected_completed VALUES (?)", (key,))
                    for status, count in metadata["status_counts"].items():
                        database.execute("INSERT INTO selected_status_totals VALUES (?, ?) ON CONFLICT(status) "
                                         "DO UPDATE SET count=count+excluded.count", (status, count))
            completed.add(key)
        summary = summarize()
        print(f"[{summary['completed_groups']}/{summary['planned_groups']}] {key}: "
              f"{metadata['attempted']} attempted, {metadata['status_counts'].get('valid', 0)} valid; "
              f"total {summary['attempted']} attempted / {summary['new_valid_grasps']} valid", flush=True)

    if args.collect_only:
        for path in sorted((output / "attempts").glob("**/epoch_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            record(metadata)
        database.close()
        return
    if initial["completed_groups"] == initial["planned_groups"]:
        print("Every configured target/taxonomy/epoch is already complete.", flush=True)
        database.close()
        return

    # Set thread limits before forkserver children import NumPy/native libraries.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import random
    import torch
    from dexonomy.op.gen_init import HandObjMatcher
    from dexonomy.data.hand_loader import HandTemplateLoader
    from dexonomy.data.obj_loader import get_object_dataloader
    from dexonomy.util.warp_util import WarpCollisionEnv

    np.random.seed(12)
    random.seed(12)
    torch.manual_seed(12)
    torch.cuda.set_device(args.gpu)
    init_cfg = OmegaConf.create(configs["init"])
    device = f"cuda:{args.gpu}"
    collision = WarpCollisionEnv(device)
    matcher = HandObjMatcher(init_cfg.op.matcher, device, collision)
    pending = set()

    def harvest(block=False):
        if not pending:
            return
        ready, _ = wait(pending, timeout=20 if block else 0, return_when=FIRST_COMPLETED)
        if not ready and block:
            print(f"Waiting for {len(pending)} native CPU batches ...", flush=True)
        for future in ready:
            pending.remove(future)
            record(future.result())

    context = multiprocessing.get_context("forkserver")
    loaders = {}
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received {signal.Signals(signum).name}: stop submitting and save "
              f"the {len(pending)} queued/running batches before exiting.", flush=True)

    previous_handlers = {signum: signal.signal(signum, request_stop)
                         for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        print(f"Starting {args.workers} native CPU workers; matcher on {device}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            for epoch in range(args.epochs):
                if stop_requested:
                    break
                obj_loader = get_object_dataloader(init_cfg.op.object, 0)
                if single_object:
                    obj_loader.dataset.path_lst = [path for path in obj_loader.dataset.path_lst
                                                   if str(Path(path).resolve()) in selected_paths]
                # Skip completed targets before loading meshes and sampling new points.
                remaining = []
                for path_string in obj_loader.dataset.path_lst:
                    if stop_requested:
                        break
                    path = Path(str(path_string))
                    unfinished = False
                    for taxonomy in taxonomies:
                        for contact_index in contact_indices(taxonomy):
                            key = job_key(path, taxonomy, epoch, contact_index)
                            if key in completed:
                                continue
                            archive_path = attempt_path(path, taxonomy, epoch, contact_index)
                            if archive_path.exists():
                                with np.load(archive_path, allow_pickle=False) as archive:
                                    metadata = json.loads(str(archive["metadata"]))
                                if metadata["run_fingerprint"] != fingerprint:
                                    raise ValueError(f"Different generation settings in {archive_path}")
                                record(metadata)
                            else:
                                unfinished = True
                    if unfinished:
                        remaining.append(path_string)
                obj_loader.dataset.path_lst = remaining
                print(f"Epoch {epoch}: {len(remaining)} unfinished targets; all types processed for each target", flush=True)
                for sample in obj_loader:
                    if stop_requested:
                        break
                    path = Path(sample["scene_path"][0])
                    rotations, translations, sampling_path = store_native_sampling(
                        output, path, epoch, fingerprint, args.points, args.inplane,
                        sample["rot_o2c_init"][0].numpy(), sample["trans_o2c_init"][0].numpy(),
                        sample["obj_scale"][0].numpy(), sample["pose_w2so"][0].numpy(),
                        source_relative=str(source_relative(path)) if single_object else None,
                    )
                    sample["rot_o2c_init"] = torch.from_numpy(rotations).unsqueeze(0)
                    sample["trans_o2c_init"] = torch.from_numpy(translations).unsqueeze(0)
                    sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                              for key, value in sample.items()}
                    collision.load(sample["col_plane"], sample["col_mesh"])
                    for taxonomy in taxonomies:
                        if stop_requested:
                            break
                        for contact_index in contact_indices(taxonomy):
                            key = job_key(path, taxonomy, epoch, contact_index)
                            if key in completed:
                                continue
                            while len(pending) >= 2 * args.workers and not stop_requested:
                                harvest(block=True)
                            if stop_requested:
                                break
                            if shutil.disk_usage(output).free < 5 * 1024 ** 3:
                                raise OSError("Output disk has less than 5 GiB free. Completed archives are preserved; "
                                              "free disk space and resume the same command.")
                            if taxonomy not in loaders:
                                loaders[taxonomy] = HandTemplateLoader(
                                    xml_path=init_cfg.hand.xml_path, skt_path=init_cfg.hand_skt_path,
                                    tmpl_name=taxonomy, init_tmpl_dir=init_cfg.init_tmpl_dir,
                                    new_tmpl_dir=init_cfg.new_tmpl_dir, max_data_buffer=init_cfg.tmpl_max_buf,
                                    n_worker=1, enumerate_contacts=all_contacts,
                                )
                            loader = loaders[taxonomy]
                            if all_contacts and loader.contact_count != contact_counts[taxonomy]:
                                raise RuntimeError(f"Hand contact count changed for {taxonomy}")
                            data_size = sample["rot_o2c_init"].shape[:2]
                            template = (loader.get_contact_batched_data(contact_index, data_size, device)
                                        if all_contacts else loader.get_batched_data(data_size, device))
                            # Adam updates translations in place. Every contact/type
                            # starts from the same saved 128*8 object surface seeds.
                            matcher_sample = {**sample, "trans_o2c_init": sample["trans_o2c_init"].clone()}
                            result = matcher.forward(template, matcher_sample)
                            inputs = {key: value.detach().cpu().numpy() for key, value in result.items()}
                            if len(inputs["grasp_pose"]) != args.points * args.inplane:
                                raise RuntimeError("Native matcher returned an unexpected candidate count")
                            inputs["grasp_qpos"] = np.concatenate([inputs["grasp_pose"], inputs["grasp_qpos"]], axis=1)
                            inputs["obj_cpn_w"] = np.concatenate([inputs["obj_cp_w"], -inputs["obj_cn_w"]], axis=-1)
                            keep = ("grasp_qpos", "obj_cpn_w", "hand_cpn_w", "ext_center", "ext_wrench", "n_evo", "obj_pose", "obj_scale")
                            job = {
                                "output": str(output), "scene_path": str(Path(sample["scene_path"][0]).resolve()),
                                "taxonomy": taxonomy, "epoch": epoch, "points": args.points, "inplane": args.inplane,
                                "run_fingerprint": fingerprint, "inputs": {key: inputs[key] for key in keep},
                                "sampling_archive": str(sampling_path), "keep_failed": args.keep_failed,
                                "hand_cbody": template["hand_cbody"], "required_cbody": template["required_cbody"],
                                "grasp_config": configs["grasp"], "eval_config": configs["eval"], "filter_config": configs["filter"],
                            }
                            if single_object:
                                job.update(single_object=True, source_relative=str(source_relative(path)),
                                           job_key=key, library_root=str(library_root))
                            if all_contacts:
                                job.update(hand_contact_index=contact_index, hand_contact_count=contact_counts[taxonomy])
                            pending.add(executor.submit(process_native_batch, job))
                            harvest()
            while pending:
                harvest(block=True)
        summary = summarize()
        outcome = "Stopped safely; resume with the same output" if stop_requested else "Finished"
        print(f"{outcome}: {summary['attempted']} attempts, {summary['new_valid_grasps']} new valid grasps", flush=True)
    finally:
        for loader in loaders.values():
            loader.stop()
        database.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main() -> None:
    args = parse_args()
    scene_root, output = resolve_path(args.scene_root), resolve_path(args.output)
    base_root = scene_root.parent / "filtered_grasps"
    if (output == scene_root or output in scene_root.parents or scene_root in output.parents
            or output == base_root or output in base_root.parents or base_root in output.parents):
        raise ValueError("Choose a separate native output directory")
    cfg_glob = str(scene_root / args.scene / f"{args.target}.npy")
    scene_paths = sorted(scene_root.glob(f"{args.scene}/{args.target}.npy"))
    if not scene_paths:
        raise ValueError(f"No target configs match {cfg_glob}")
    taxonomies = select_taxonomies(args, base_root)
    print(f"Targets: {len(scene_paths)}; taxonomies: {len(taxonomies)}", flush=True)
    if args.save_all:
        print(f"All {args.points * args.inplane} candidates per target/type/epoch; "
              f"{len(scene_paths) * len(taxonomies) * args.epochs * args.points * args.inplane} total attempts", flush=True)
        if args.dry_run:
            run_all_candidates(args, scene_paths, base_root, output, taxonomies, cfg_glob)
        else:
            with output_lock(output):
                (output / "generation.pid").write_text(str(os.getpid()) + "\n")
                runtime_path = output / "runtime.json"
                runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
                if runtime.get("pid") != os.getpid():
                    runtime.update(previous_pid=runtime.get("pid"),
                                   previous_started_at=runtime.get("started_at"),
                                   started_at=datetime.now(timezone.utc).isoformat())
                runtime.update(pid=os.getpid(), workers=args.workers, gpu=args.gpu,
                               keep_failed=args.keep_failed,
                               command=[sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]])
                runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
                run_all_candidates(args, scene_paths, base_root, output, taxonomies, cfg_glob)
        return
    if not args.collect_only:
        print(f"Up to {len(scene_paths) * len(taxonomies) * args.epochs * (args.candidates or 20)} initializations", flush=True)
    if not args.collect_only:
        for taxonomy in taxonomies:
            for stage in ("init", "grasp", "eval"):
                command = native_command(args, output, cfg_glob, taxonomy, stage)
                if args.dry_run:
                    print(shlex.join(command))
                else:
                    run_native(command, output, taxonomy, stage)
    if not args.dry_run:
        collect_results(scene_paths, base_root, output)


if __name__ == "__main__":
    main()
