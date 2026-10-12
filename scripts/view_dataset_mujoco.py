#!/usr/bin/env python3
"""Play released Dexonomy grasp samples sequentially in MuJoCo Viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_VisEnv
from dexonomy.util.file_util import load_scene_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Dexonomy dataset samples sequentially with MuJoCo."
    )
    parser.add_argument(
        "--root",
        default="data/extracted/Dexonomy_GRASP_shadow/succ_collect",
        help="Root directory containing taxonomy folders.",
    )
    parser.add_argument("--taxonomy", default="1_Large_Diameter")
    parser.add_argument(
        "--filter",
        default="",
        help="Only include dataset paths containing this substring.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of dataset files to show; 0 means all.",
    )
    parser.add_argument(
        "--qpos",
        choices=("grasp", "pregrasp", "squeeze", "traj_state", "traj_ctrl"),
        default="grasp",
        help="Pose sequence to display from each dataset file.",
    )
    parser.add_argument(
        "--pose-seconds",
        type=float,
        default=0.5,
        help="Seconds per pose. Set 0 to start paused.",
    )
    parser.add_argument("--distance", type=float, default=0.7)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument(
        "--hand-xml", default="assets/hand/shadow/right.xml"
    )
    return parser.parse_args()


def collect_paths(args: argparse.Namespace) -> tuple[list[Path], int]:
    taxonomy_dir = Path(args.root) / args.taxonomy
    if not taxonomy_dir.is_dir():
        raise FileNotFoundError(f"Taxonomy directory does not exist: {taxonomy_dir}")

    all_paths = sorted(taxonomy_dir.rglob("*.npy"))
    if args.filter:
        all_paths = [path for path in all_paths if args.filter in str(path)]
    total = len(all_paths)

    start = max(0, args.start)
    paths = all_paths[start:]
    if args.limit > 0:
        paths = paths[: args.limit]
    return paths, total


def load_sample(
    path: Path, args: argparse.Namespace
) -> tuple[MuJoCo_VisEnv, np.ndarray]:
    sample = np.load(path, allow_pickle=True).item()
    qpos_key = f"{args.qpos}_qpos"
    if qpos_key not in sample:
        raise KeyError(f"{path} does not contain {qpos_key}")
    if "scene_path" not in sample:
        raise KeyError(f"{path} does not contain scene_path")

    qposes = np.asarray(sample[qpos_key])
    if qposes.ndim == 1:
        qposes = qposes[None]
    if len(qposes) == 0:
        raise ValueError(f"{path} contains an empty {qpos_key}")

    env = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=args.hand_xml, freejoint=True),
        scene_cfg=load_scene_cfg(sample["scene_path"]),
        sim_cfg=MuJoCo_OptCfg(),
        vis_mode="visual",
    )
    return env, qposes


def set_pose(env: MuJoCo_VisEnv, qpos: np.ndarray) -> None:
    env.forward_kinematics(qpos)
    mujoco.mj_forward(env._model, env._data)


def show_sample(
    path: Path,
    display_index: int,
    display_total: int,
    args: argparse.Namespace,
) -> tuple[int, bool]:
    env, qposes = load_sample(path, args)
    pose_index = 0
    set_pose(env, qposes[pose_index])

    state = {
        "sample_delta": 0,
        "pose_delta": 0,
        "quit": False,
        "paused": args.pose_seconds <= 0,
    }

    def on_key(key: int) -> None:
        if key in (glfw.KEY_RIGHT, glfw.KEY_D):
            state["sample_delta"] = 1
        elif key in (glfw.KEY_LEFT, glfw.KEY_A):
            state["sample_delta"] = -1
        elif key == glfw.KEY_N:
            state["pose_delta"] = 1
        elif key == glfw.KEY_P:
            state["pose_delta"] = -1
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            state["quit"] = True

    print(
        f"[{display_index + 1}/{display_total}] {path} "
        f"({len(qposes)} {args.qpos} poses)",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        env._model,
        env._data,
        key_callback=on_key,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = qposes[0][:3]
        viewer.cam.distance = args.distance
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        next_pose_time = time.monotonic() + max(args.pose_seconds, 0.0)

        while viewer.is_running():
            if state["quit"]:
                return 0, True
            if state["sample_delta"]:
                return int(state["sample_delta"]), False

            pose_delta = int(state["pose_delta"])
            state["pose_delta"] = 0
            if pose_delta:
                new_pose_index = pose_index + pose_delta
                if new_pose_index < 0:
                    return -1, False
                if new_pose_index >= len(qposes):
                    return 1, False
                pose_index = new_pose_index
                set_pose(env, qposes[pose_index])
                next_pose_time = time.monotonic() + max(args.pose_seconds, 0.0)
                print(f"  pose {pose_index + 1}/{len(qposes)}", flush=True)

            now = time.monotonic()
            if (
                not state["paused"]
                and args.pose_seconds > 0
                and now >= next_pose_time
            ):
                pose_index += 1
                if pose_index >= len(qposes):
                    return 1, False
                set_pose(env, qposes[pose_index])
                next_pose_time = now + args.pose_seconds
                print(f"  pose {pose_index + 1}/{len(qposes)}", flush=True)

            viewer.sync()
            time.sleep(0.01)

    return 1, False


def main() -> int:
    args = parse_args()
    try:
        paths, total = collect_paths(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if not paths:
        print("No matching dataset samples found.", file=sys.stderr)
        return 1

    print(f"Found {total} matching samples; selected {len(paths)}.")
    print("Keys: Right/D next sample | Left/A previous sample | N/P pose")
    print("      Space pause/resume | Q/Esc quit")

    index = 0
    while 0 <= index < len(paths):
        try:
            delta, should_quit = show_sample(
                paths[index], index, len(paths), args
            )
        except Exception as exc:
            print(f"Skip {paths[index]}: {type(exc).__name__}: {exc}", file=sys.stderr)
            delta, should_quit = 1, False

        if should_quit:
            break
        index = max(0, index + delta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
