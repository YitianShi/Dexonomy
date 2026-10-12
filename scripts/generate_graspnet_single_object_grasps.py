#!/usr/bin/env python3
"""GraspNet-1Billion single-object grasp generation and physics validation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_single_object_grasps import main as generate


def main():
    generate(dataset="graspnet")


if __name__ == "__main__":
    main()
