#!/usr/bin/env python3
"""Generate a reusable, isolated object/scale grasp library with native Dexonomy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dexonomy.op.native_clutter_batch import atomic_npy
from dexonomy.util.file_util import load_scene_cfg
from scripts.generate_native_clutter_grasps import (
    output_lock, resolve_path, run_all_candidates, select_taxonomies, template_contact_counts,
)

DATASET_PRESETS = {
    "graspnet": {
        "label": "GraspNet-1Billion",
        "object_root": REPO_ROOT / "assets/object/graspnet_1billion/scene_cfg",
        "output": REPO_ROOT / "output/single_object_grasps_graspnet_shadow",
        "object": "000",
    },
    "dexgraspnet": {
        "label": "DexGraspNet",
        "object_root": REPO_ROOT / "assets/object/DGN_5k/scene_cfg",
        "output": REPO_ROOT / "output/single_object_grasps_dgn_shadow",
        "object": "core_bottle_1071fa4cddb2da2fc8724d5673a063a6",
    },
}


def canonical_object_scene(scene):
    """Discard scene pose, table and distractors; key only the target asset and scale."""
    target = scene["task"]["obj_name"]
    original = scene["scene"][target]
    if original["type"] != "rigid_object":
        raise ValueError(f"Single-object library supports rigid objects: {target}")
    if Path(target).name != target:
        raise ValueError(f"Object name must be one path component: {target}")
    scale = np.asarray(original["scale"], dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Invalid physical object scale: {target}")
    asset = {key: str(Path(original[key]).resolve())
             for key in ("file_path", "xml_path", "urdf_path", "info_path") if key in original}
    identity = {"object": target, "asset": asset, "scale": scale.tolist()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    relative = Path(target) / "floating" / f"scale_{digest}.npy"
    canonical = {
        "scene": {target: {"type": "rigid_object", **asset, "scale": scale,
                            "pose": np.asarray([0., 0., 0., 1., 0., 0., 0.])}},
        "scene_id": str(relative.with_suffix("")),
        "task": {"type": "force_closure", "obj_name": target},
    }
    return relative, canonical


def collect_objects(args):
    selected_names = None
    if args.object_inventory:
        inventory = resolve_path(args.object_inventory)
        selected_names = set()
        for path in sorted(inventory.glob("scene_*/*.npy")):
            scene = load_scene_cfg(str(path))
            selected_names.update(name for name, body in scene["scene"].items()
                                  if body["type"] == "rigid_object")
        selected_names = {name for name in selected_names if fnmatch.fnmatch(name, args.object)}
        if not selected_names:
            raise ValueError(f"No selected rigid objects in scene inventory: {inventory}")
        for name in selected_names:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"Object name must be one path component: {name}")
    # A clutter inventory supplies only asset identities and physical sizes. Its
    # obstacle geometry and poses never enter generation or physical validation.
    if args.scale_inventory:
        root = resolve_path(args.scale_inventory)
        paths = sorted(root.glob(f"{args.scene}/*.npy"))
    else:
        root = resolve_path(args.object_root)
        if selected_names is None:
            paths = sorted(root.glob(f"{args.object}/floating/*.npy"))
        else:
            paths = []
            for name in sorted(selected_names):
                object_paths = sorted((root / name / "floating").glob("*.npy"))
                if not object_paths:
                    raise ValueError(f"Scene object has no floating asset configurations: {name}")
                paths.extend(object_paths)
    catalog = {}
    for path in paths:
        scene = load_scene_cfg(str(path))
        name = scene["task"]["obj_name"]
        if (fnmatch.fnmatch(name, args.object) and
                (selected_names is None or name in selected_names)):
            relative, isolated = canonical_object_scene(scene)
            catalog.setdefault(relative, isolated)
    if not catalog:
        raise ValueError(f"No object/scale configurations selected from {root}")
    return catalog


def parse_args(dataset=None):
    preset = DATASET_PRESETS[dataset] if dataset is not None else None
    parser = argparse.ArgumentParser(description=(
        f"Generate isolated Shadow-hand grasps for {preset['label']}. Asset and output paths are fixed."
        if preset else __doc__))
    if preset:
        parser.set_defaults(object_root=str(preset["object_root"]), output=str(preset["output"]),
                            object_inventory=None, scale_inventory=None, scene="scene_*")
    else:
        parser.add_argument("--object-root", default="assets/object/DGN_5k/scene_cfg")
        parser.add_argument("--object-inventory", help="Select only rigid object IDs present in scene_*/ configs; poses and scales are not imported")
        parser.add_argument("--scale-inventory", help="Optional clutter scene_cfg directory used only to select unique object sizes")
        parser.add_argument("--scene", default="scene_*", help="Inventory scene name or glob")
        parser.add_argument("--output", default="output/single_object_grasps_dgn_shadow")
    parser.add_argument("--object", default=preset["object"] if preset else "*",
                        help="Object asset ID or glob; use '*' to process all objects independently")
    parser.add_argument("--taxonomies", nargs="+")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--inplane", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=84)
    parser.add_argument("--keep-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--all-hand-contacts", action=argparse.BooleanOptionalAction, default=True,
                        help="Enumerate every hand contact against all object points and rolls")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.points, args.inplane, args.workers) < 1 or args.gpu < 0:
        parser.error("epochs, points, inplane and workers must be positive; gpu must be nonnegative")
    return args


def generation_directory(output, args):
    """Keep checkpoints for different sampling settings separate in one library."""
    if getattr(args, "all_hand_contacts", False):
        return output / "runs" / f"all_contacts_points_{args.points}_inplane_{args.inplane}_epochs_{args.epochs}"
    settings_path = output / "run_config.json"
    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if any(saved.get(key) != getattr(args, key) for key in ("points", "inplane", "epochs")):
            return output / "runs" / f"points_{args.points}_inplane_{args.inplane}_epochs_{args.epochs}"
    return output


def main(dataset=None):
    args = parse_args(dataset) if dataset is not None else parse_args()
    output = resolve_path(args.output)
    for value in filter(None, (args.scale_inventory, args.object_root, args.object_inventory)):
        source = resolve_path(value)
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Object library output must be separate from input scene configurations")
    catalog = collect_objects(args)
    args.save_all = True
    args.collect_only = False
    args.single_object = True
    args.candidates = None
    taxonomies = select_taxonomies(args, output / "succ_collect")
    all_contacts = getattr(args, "all_hand_contacts", False)
    contact_counts = template_contact_counts(taxonomies) if all_contacts else {name: 1 for name in taxonomies}
    groups = len(catalog) * sum(contact_counts.values()) * args.epochs
    print(json.dumps({"stage": "single_object", "objects": len({p.parts[0] for p in catalog}),
                      "unique_object_scales": len(catalog), "taxonomy_count": len(taxonomies),
                      "hand_contact_mode": "all" if all_contacts else "random",
                      "hand_contact_counts": contact_counts if all_contacts else None,
                      "candidates_per_hand_contact": args.points * args.inplane,
                      "planned_groups": groups,
                      "planned_attempts": groups * args.points * args.inplane,
                      "skip_traj": True, "generation_environment": "one floating object at origin",
                      "grasp_library": str(output / "succ_collect")}, indent=2), flush=True)
    if args.dry_run:
        return
    with output_lock(output):
        run_output = generation_directory(output, args)
        run_output.mkdir(parents=True, exist_ok=True)
        args.library_root = str(output)
        scene_root = output / "scene_cfg"
        for relative, scene in catalog.items():
            path = scene_root / relative
            if path.exists():
                existing = load_scene_cfg(str(path))
                existing_relative, _ = canonical_object_scene(existing)
                target = existing["task"]["obj_name"]
                if (existing_relative != relative or set(existing["scene"]) != {target} or
                        not np.array_equal(existing["scene"][target]["pose"], scene["scene"][target]["pose"])):
                    raise ValueError(f"Existing isolated scene differs: {path}")
            else:
                atomic_npy(path, scene)
        # Preserve cached assets, but generate only the current selection.
        paths = [scene_root / relative for relative in sorted(catalog)]
        args.scene_root, args.scene, args.target = str(scene_root), "*/floating", "*"
        runtime = {"pid": os.getpid(), "stage": "single_object", "workers": args.workers,
                   "gpu": args.gpu, "skip_traj": True, "keep_failed": args.keep_failed,
                   "points": args.points, "inplane": args.inplane,
                   "hand_contact_mode": "all" if all_contacts else "random",
                   "run_directory": str(run_output), "run_config": str(run_output / "run_config.json"),
                   "object_inventory": (str(resolve_path(args.object_inventory))
                                        if args.object_inventory else None),
                   "state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                   "command": [sys.executable, *sys.argv]}
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        try:
            run_all_candidates(args, paths, output / "unused_baseline", run_output, taxonomies,
                               str(scene_root / "*" / "floating" / "*.npy"))
            progress = json.loads((run_output / "progress.json").read_text())
            runtime["state"] = ("complete" if progress["completed_groups"] == progress["planned_groups"] else "stopped")
        except BaseException as exc:
            runtime.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            runtime["updated_at"] = datetime.now(timezone.utc).isoformat()
            (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")


if __name__ == "__main__":
    main()
