#!/usr/bin/env python3
"""Filter the reusable GraspNet grasp library in the 450 official test scenes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_graspnet_scene_grasps import main as filter_scenes


def main(argv=None):
    return filter_scenes([
        "--scene-root", "assets/scene/dexgraspnet2_graspnet_test/scene_cfg",
        "--models", "assets/object/graspnet_1billion",
        "--library", "output/datasets/single_object/graspnet",
        "--output", "work_runs/graspnet_test_scene_filter_shadow",
        "--scene", "scene_*", "--workers", "84",
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
