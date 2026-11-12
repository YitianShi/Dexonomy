#!/usr/bin/env python3
"""Filter the 937-object grasp library in all 219 official ShapeNet test scenes."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_graspnet_scene_grasps import main as filter_scenes


def main(argv=None):
    return filter_scenes([
        "--scene-root", "assets/scene/dexgraspnet2_shapenet_test/scene_cfg",
        "--models", "assets/object/dexgraspnet2_shapenet",
        "--library", "output/datasets/single_object/shapenet",
        "--output", "work_runs/dexgraspnet2_scene_filter_shadow",
        "--scene", "scene_*", "--workers", "84",
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
