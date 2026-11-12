"""Dataset-viewer commands and readable object/scale metadata.

The Qt dependency is loaded only when opening the desktop panel.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ALL_TYPES = "全部抓取类型"
DATASET_LABELS = {
    "graspnet_single": "GraspNet · 单物体",
    "shapenet_single": "ShapeNet · 单物体",
    "graspnet_clutter": "GraspNet · 训练场景",
    "shapenet_clutter": "ShapeNet · 测试场景",
    "graspnet_test_clutter": "GraspNet · 测试场景",
}
DATASET_LABELS.update({
    'taxonomy_filtered_'+key: value+' · 接触复验'
    for key,value in list(DATASET_LABELS.items())
})
DATASET_LABELS.update({
    'deduplicated_'+key: value+' · 接触复验＋去重'
    for key,value in list(DATASET_LABELS.items())
    if not key.startswith('taxonomy_filtered_')
})
SAMPLING = {"FPS · 最远点采样": "fps", "随机采样": "random"}


def group_taxonomies(group) -> set[str]:
    if group.samples is not None:
        return {taxonomy for taxonomy, _ in group.samples}
    return {directory.parent.name for directory in group.directories}


def build_viewer_command(args, root, item, taxonomy, count, seed, sampling, phase, alpha):
    count, seed, alpha = int(count), int(seed), float(alpha)
    if count < 0 or seed < 0:
        raise ValueError("显示数量和随机种子必须是非负整数。")
    if not 0 <= alpha <= 1:
        raise ValueError("手部透明度必须在 0 到 1 之间。")
    if sampling not in SAMPLING.values() or phase not in ("pregrasp", "grasp", "squeeze"):
        raise ValueError("请选择有效的采样方法和姿态阶段。")
    if not item:
        raise ValueError("请先选择物体／尺度或场景。")
    return [
        sys.executable, str(Path(__file__).with_name("view_clutter_grasps_mujoco.py")),
        "--root", str(root), "--item", item, "--taxonomy", taxonomy,
        "--max-grasps", str(count), "--sample-seed", str(seed),
        "--sampling", sampling, "--phase", phase, "--hand-alpha", str(alpha),
        "--hand-xml", args.hand_xml,
        "--distance", str(args.distance), "--azimuth", str(args.azimuth),
        "--elevation", str(args.elevation),
    ]


def scale_description(scale) -> str:
    """Describe actual model-space scaling, never a hashed sample filename."""
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size != 3 or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f"Invalid object scale: {scale}")
    if np.allclose(scale, 1, rtol=0, atol=1e-8):
        return "原始尺度 · 1.00×"
    if np.allclose(scale, scale[0], rtol=0, atol=1e-8):
        return f"等比缩放 · {scale[0]:.4g}×"
    return "非等比缩放 · " + " / ".join(
        f"{axis} {value:.4g}×" for axis, value in zip("XYZ", scale)
    )


def read_group_details(group):
    """Read one selection's real scale and mesh bounds in metres, off the UI thread."""
    import trimesh
    from dexonomy.util.file_util import load_scene_cfg

    if group.samples is not None:
        sample_path = group.samples[0][1]
    else:
        sample_path = next(path for directory in group.directories
                           for path in sorted(directory.glob("*.npy")))
    sample = np.load(sample_path, allow_pickle=True).item()
    scene_path = Path(str(sample["scene_path"])).expanduser()
    if not scene_path.is_file() and not scene_path.is_absolute():
        for base in (sample_path.parent, Path(__file__).resolve().parents[1]):
            if (base / scene_path).is_file():
                scene_path = base / scene_path
                break
    scene = load_scene_cfg(str(scene_path))
    objects = [(name, cfg) for name, cfg in scene["scene"].items()
               if cfg.get("type") == "rigid_object"]
    geometries = []
    object_dimensions = {}
    scales = {}
    for name, cfg in objects:
        scale = np.asarray(cfg["scale"], dtype=float)
        scale_description(scale)  # Validate before displaying or transforming.
        scales[name] = scale
        mesh = trimesh.load(cfg["file_path"], force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=float) * scale
        object_dimensions[name] = np.ptp(vertices, axis=0)
        pose = np.asarray(cfg.get("pose", [0, 0, 0, 1, 0, 0, 0]), dtype=float)
        transform = trimesh.transformations.quaternion_matrix(pose[3:])
        vertices = vertices @ transform[:3, :3].T + pose[:3]
        geometries.append((vertices, np.asarray(mesh.faces, dtype=np.int32)))
    if not geometries:
        raise ValueError("这个条目没有可预览的物体模型。")
    single = group.samples is not None
    target = str(sample.get("target_name", objects[0][0]))
    if single and target not in scales:
        raise ValueError(f"Target {target!r} missing from scene config")
    all_vertices = np.concatenate([vertices for vertices, _ in geometries])
    dimensions = (object_dimensions[target] if single
                  else np.ptp(all_vertices, axis=0))
    return {
        "name": group.name, "single": single, "target": target,
        "scale": scales.get(target) if single else None,
        "scale_label": (scale_description(scales[target]) if single
                        else "使用场景中记录的各物体尺度"),
        "dimensions": dimensions, "geometries": geometries,
        "object_count": len(objects), "scene_path": str(scene_path),
    }


def launch_panel(args, collect_groups) -> int:
    try:
        if __package__:
            from .grasp_viewer_qt import launch_panel as launch_qt
        else:
            from grasp_viewer_qt import launch_panel as launch_qt
    except ImportError as exc:
        raise RuntimeError(
            "控制面板需要 Qt。请运行 python -m pip install -e '.[viewer]'。"
            f"\n{exc}"
        ) from exc
    return launch_qt(args, collect_groups)
