#!/usr/bin/env python3
"""Generate isolated grasps for DexGraspNet 2.0's released ShapeNet test objects."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_single_object_grasps import main as generate


def main():
    generate(dataset="dexgraspnet2")


if __name__ == "__main__":
    main()
