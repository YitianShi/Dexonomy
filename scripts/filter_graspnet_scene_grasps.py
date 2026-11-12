#!/usr/bin/env python3
"""Filter reusable GraspNet hand grasps in the imported official scene layouts."""

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
import shutil
import signal
import sqlite3
import sys
import time
from tempfile import mkdtemp

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTS = ("input_count", "evaluated_count", "valid_count", "rejected_scale",
          "rejected_collision", "rejected_target_penetration",
          "rejected_target_contact", "rejected_lift")
_SOURCE_SCENES = {}


def resolve_path(value):
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", default="assets/scene/dexgraspnet2_train/scene_cfg")
    parser.add_argument("--library", default="output/datasets/single_object/graspnet")
    parser.add_argument("--models", default="assets/object/graspnet_1billion",
                        help="Imported object assets used by the selected scene/library.")
    parser.add_argument("--output", default="work_runs/graspnet_scene_filter_shadow")
    parser.add_argument("--scene", default="scene_0000", help="Scene name/glob; scene_* selects all 7,600.")
    parser.add_argument("--target", default="*", help="Object ID/glob within the selected scenes.")
    parser.add_argument("--taxonomies", nargs="+", help="Default: all 33 Shadow hand templates.")
    parser.add_argument("--workers", type=int, default=24, help="CPU processes; no GPU is used.")
    parser.add_argument("--approach-steps", type=int, default=8)
    parser.add_argument("--max-per-target", type=int, default=0,
                        help="Candidate cap per target/taxonomy for debugging; 0 checks all.")
    parser.add_argument("--check-pregrasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lift", action=argparse.BooleanOptionalAction, default=False,
                        help="Also simulate closing and lifting in the complete scene.")
    parser.add_argument("--dry-run", action="store_true", help="Show selection/configuration without writing or filtering.")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.approach_steps < 1 or args.max_per_target < 0:
        parser.error("workers/approach-steps must be positive; max-per-target must be nonnegative")
    for name in ("scene", "target"):
        value = getattr(args, name)
        if "/" in value or "\\" in value or value in {".", ".."}:
            parser.error(f"--{name} must be a name or glob, without directory components")
    return args


def file_token(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return [str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns]
    except FileNotFoundError:
        return [str(path), None]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Scene filtering is already running in {output}") from None
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            fcntl.flock(lock, fcntl.LOCK_UN)


def select_inputs(args):
    scene_root, library, output = map(resolve_path, (args.scene_root, args.library, args.output))
    if not scene_root.is_dir() or not library.is_dir():
        raise FileNotFoundError(f"Scene configs/library must exist: {scene_root}, {library}")
    for source in (scene_root, library):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Filtering output must be separate from its inputs: {output}")
    paths = sorted(path.resolve() for folder in scene_root.glob(args.scene) if folder.is_dir()
                   for path in folder.glob(f"{args.target}.npy") if path.is_file())
    if not paths:
        raise ValueError("No scene targets matched the selection")
    templates = REPO_ROOT / "assets/hand/shadow/init_tmpl"
    taxonomies = args.taxonomies or [path.stem for path in templates.glob("[0-9]*.npy")]
    for name in taxonomies:
        if Path(name).name != name or not (templates / f"{name}.npy").is_file():
            raise ValueError(f"Unknown Shadow hand taxonomy: {name}")
    taxonomies = sorted(set(taxonomies), key=lambda name: int(name.split("_", 1)[0]))
    return scene_root, library, output, paths, taxonomies


def filter_config(args, library, taxonomies):
    # Reuse the existing CPU MuJoCo filter and its contact/lift thresholds.
    cfg = OmegaConf.load(REPO_ROOT / "dexonomy/config/op/clutter.yaml").filter
    cfg.source_grasp_dir = str(library)
    cfg.source_scene_dir = str(resolve_path(args.models) / "scene_cfg")
    cfg.scene_cfg_glob = ""  # Selection is handled by this independent entry.
    cfg.output_dir = ""  # Workers write to staging; the parent publishes stable results.
    cfg.hand_xml = str(REPO_ROOT / "assets/hand/shadow/right.xml")
    cfg.single_object_library = True
    cfg.strict_source = True
    cfg.short_circuit_collisions = True
    cfg.taxonomies = taxonomies
    cfg.max_per_target = args.max_per_target
    cfg.approach_steps = args.approach_steps
    cfg.check_pregrasp = args.check_pregrasp
    cfg.lift.enabled = args.lift
    return OmegaConf.to_container(cfg, resolve=True)


def implementation_signature(models=None):
    paths = [Path(__file__), REPO_ROOT / "dexonomy/op/gen_clutter.py",
             REPO_ROOT / "dexonomy/sim/mujoco_env.py", REPO_ROOT / "dexonomy/sim/basic.py",
             REPO_ROOT / "dexonomy/util/np_util.py", REPO_ROOT / "dexonomy/util/file_util.py",
             REPO_ROOT / "dexonomy/op/native_clutter_batch.py"]
    code = [[str(path), hashlib.sha256(path.read_bytes()).hexdigest()]
            for path in paths if path.is_file()]
    roots = [REPO_ROOT / "assets/hand/shadow",
             REPO_ROOT / "third_party/mujoco_menagerie/shadow_hand/assets",
             resolve_path(models or "assets/object/graspnet_1billion") / "processed_data"]
    assets = [file_token(path) for root in roots for path in sorted(root.rglob("*")) if path.is_file()]
    import mujoco
    return digest([code, assets, mujoco.__version__, np.__version__])


def input_signature(scene_path, cfg):
    """Include atomic library replacements, additions, removals and source poses."""
    inputs = [file_token(scene_path)]
    for taxonomy in cfg["taxonomies"]:
        folder = Path(cfg["source_grasp_dir"]) / taxonomy / Path(scene_path).stem / "floating"
        files = sorted(folder.glob("*.npy"))
        inputs.append([taxonomy, [str(path) for path in files]])
        for path in files:
            token = file_token(path)
            key = str(path)
            cached = _SOURCE_SCENES.get(key)
            if cached is None or cached[0] != token:
                data = np.load(path, allow_pickle=True).item()
                source_scene = resolve_path(str(data["scene_path"])) if "scene_path" in data else None
                _SOURCE_SCENES[key] = (token, source_scene)
            source_scene = _SOURCE_SCENES[key][1]
            inputs.append([token, file_token(source_scene) if source_scene is not None else None])
    return digest(inputs)


def filter_target(scene_path, cfg, staging_root):
    """One process loads one full static scene, then checks all selected templates."""
    from dexonomy.op.gen_clutter import _filter_scene

    before = input_signature(scene_path, cfg)
    staging = Path(mkdtemp(prefix="target_", dir=staging_root))
    worker_cfg = dict(cfg, output_dir=str(staging))
    try:
        records = _filter_scene(str(scene_path), OmegaConf.create(worker_cfg))
        after = input_signature(scene_path, cfg)
        return {"signature": before, "stable": before == after,
                "staging": str(staging), "records": records}
    except BaseException:
        shutil.rmtree(staging)
        raise


def publish_result(scene_path, result, filtered_root, cfg):
    """Do not publish/checkpoint results produced while a library file was changing."""
    staging = Path(result["staging"])
    try:
        if not result["stable"] or result["signature"] != input_signature(scene_path, cfg):
            return None
        records = result["records"]
        if any(record["status"] == "invalid_source_grasp" for record in records):
            raise ValueError(f"Invalid source grasps for {scene_path}")
        for record in records:
            final = filtered_root / record["taxonomy"] / Path(scene_path).parent.name / Path(scene_path).name
            if record.get("valid_count", 0):
                final.parent.mkdir(parents=True, exist_ok=True)
                Path(record["output_path"]).replace(final)
                record["output_path"] = str(final)
                record["output_token"] = file_token(final)
            else:
                final.unlink(missing_ok=True)
                record["output_path"] = None
        return records
    finally:
        shutil.rmtree(staging)


def checkpoint_valid(row, signature):
    if row is None or row[0] != signature:
        return None
    records = json.loads(row[1])
    for record in records:
        if record.get("valid_count", 0):
            if file_token(record["output_path"]) != record.get("output_token"):
                return None
    return records


def discard_target_outputs(filtered_root, scene_path, taxonomies):
    """Exclude stale derived results while a target is awaiting revalidation."""
    for taxonomy in taxonomies:
        path = filtered_root / taxonomy / Path(scene_path).parent.name / Path(scene_path).name
        path.unlink(missing_ok=True)


def run_filter(args, scene_root, library, output, paths, taxonomies, cfg):
    # Settings/implementation changes get their own result directory.
    run_id = digest([cfg, implementation_signature(args.models)])[:16]
    run_root = output / "runs" / run_id
    filtered_root, staging_root = run_root / "filtered_grasps", run_root / ".staging"
    with output_lock(output):
        staging_root.mkdir(parents=True, exist_ok=True)
        for path in staging_root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)  # An interrupted worker's unpublished output.
        atomic_json(run_root / "settings.json", cfg)
        db = sqlite3.connect(run_root / "progress.sqlite")
        db.execute("CREATE TABLE IF NOT EXISTS completed (job_key TEXT PRIMARY KEY, signature TEXT, records TEXT)")
        counts, statuses = Counter(), Counter()
        finished = resumed = changed = failed = 0
        started = time.monotonic()
        last_progress = started
        pending = {}
        iterator = iter(paths)
        exhausted = False

        def progress(stage):
            nonlocal last_progress
            value = {"stage": stage, "pid": os.getpid(), "run_id": run_id,
                     "updated_at": datetime.now(timezone.utc).isoformat(),
                     "scene_root": str(scene_root), "library": str(library),
                     "filtered_grasps": str(filtered_root),
                     "scenes": len({path.parent.name for path in paths}),
                     "targets": len(paths), "taxonomies": len(taxonomies),
                     "finished_targets": finished, "resumed_targets": resumed,
                     "active_targets": len(pending), "changed_input_targets": changed,
                     "failed_targets": failed, "taxonomy_statuses": dict(statuses),
                     "counts": {key: counts[key] for key in COUNTS},
                     "elapsed_seconds": round(time.monotonic() - started, 1),
                     "workers": min(args.workers, len(paths)), "gpu": False,
                     "check_pregrasp": args.check_pregrasp, "lift": args.lift}
            atomic_json(output / "progress.json", value)
            atomic_json(run_root / "progress.json", value)
            last_progress = time.monotonic()
            return value

        def report_error(key, exc):
            error = {"target": key, "error": str(exc),
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            with (run_root / "errors.jsonl").open("a") as stream:
                stream.write(json.dumps(error) + "\n")
            print(f"ERROR {key}: {exc}", file=sys.stderr, flush=True)

        def account(records):
            for record in records:
                statuses[record["status"]] += 1
                counts.update({key: record.get(key, 0) for key in COUNTS})

        # Spawn avoids inheriting CUDA contexts and unsafe native thread state.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        executor = ProcessPoolExecutor(max_workers=min(args.workers, len(paths)),
                                       mp_context=multiprocessing.get_context("spawn"))
        try:
            progress("running")
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * args.workers:
                    if time.monotonic() - last_progress >= 10:
                        progress("running")
                    try:
                        path = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    key = str(path.relative_to(scene_root))
                    try:
                        signature = input_signature(path, cfg)
                        row = db.execute("SELECT signature, records FROM completed WHERE job_key=?", (key,)).fetchone()
                        records = checkpoint_valid(row, signature)
                        if records is not None:
                            account(records)
                            finished += 1
                            resumed += 1
                            continue
                        # Never let a stale checkpoint represent changed inputs.
                        db.execute("DELETE FROM completed WHERE job_key=?", (key,))
                        db.commit()
                        discard_target_outputs(filtered_root, path, taxonomies)
                        future = executor.submit(filter_target, str(path), cfg, str(staging_root))
                        pending[future] = (path, key)
                    except Exception as exc:
                        failed += 1
                        db.execute("DELETE FROM completed WHERE job_key=?", (key,))
                        db.commit()
                        discard_target_outputs(filtered_root, path, taxonomies)
                        report_error(key, exc)
                if not pending:
                    continue
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    path, key = pending.pop(future)
                    try:
                        result = future.result()
                        records = publish_result(path, result, filtered_root, cfg)
                        if records is None:
                            changed += 1
                            print(f"INPUT CHANGED {key}: deferred; rerun to filter the new library", flush=True)
                            continue
                        db.execute("INSERT OR REPLACE INTO completed VALUES (?, ?, ?)",
                                   (key, result["signature"], json.dumps(records)))
                        db.commit()
                        account(records)
                        finished += 1
                        print(f"{finished}/{len(paths)} targets; {key}: "
                              f"{sum(r.get('valid_count', 0) for r in records)} retained", flush=True)
                    except Exception as exc:
                        failed += 1
                        db.execute("DELETE FROM completed WHERE job_key=?", (key,))
                        db.commit()
                        discard_target_outputs(filtered_root, path, taxonomies)
                        report_error(key, exc)
                progress("running")
            stage = "incomplete" if changed or failed else "completed_snapshot"
            result = progress(stage)
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 1 if changed or failed else 0
        except BaseException:
            progress("interrupted")
            for process in list(executor._processes.values()):
                process.terminate()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            db.close()


def main(argv=None):
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    args = parse_args(argv)
    scene_root, library, output, paths, taxonomies = select_inputs(args)
    cfg = filter_config(args, library, taxonomies)
    print(json.dumps({"scenes": len({path.parent.name for path in paths}), "targets": len(paths),
                      "taxonomies": taxonomies, "workers": min(args.workers, len(paths)),
                      "output": str(output), "filter": cfg}, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    return run_filter(args, scene_root, library, output, paths, taxonomies, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
