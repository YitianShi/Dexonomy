#!/usr/bin/env python3
"""Audit completed object-scale grasps without changing the source library."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import zipfile
import csv

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from transforms3d.quaternions import quat2mat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dexonomy.util.grasp_diversity import TOLERANCES, analyze_grasps
from scripts.generate_native_clutter_grasps import output_lock, resolve_path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed_groups(run, object_pattern="*"):
    settings = json.loads((run / "run_config.json").read_text())
    expected = {(taxonomy, epoch) for taxonomy in settings["taxonomies"]
                for epoch in range(settings["epochs"])}
    done = defaultdict(set)
    database = sqlite3.connect(f"file:{run / 'progress.sqlite'}?mode=ro", uri=True)
    try:
        for (key,) in database.execute("SELECT job_key FROM completed"):
            taxonomy, remainder = key.split("/", 1)
            relative, epoch = remainder.rsplit("/", 1)
            if fnmatch.fnmatch(relative.split("/")[0], object_pattern):
                done[relative].add((taxonomy, int(epoch)))
    finally:
        database.close()
    return sorted(relative for relative, types in done.items() if expected <= types), settings


def load_group(library, relative, taxonomies, phase):
    scene_path = library / "scene_cfg" / f"{relative}.npy"
    scene = np.load(scene_path, allow_pickle=True).item()
    name = scene["task"]["obj_name"]
    body = scene["scene"][name]
    if set(scene["scene"]) != {name}:
        raise ValueError("Diversity audit expects isolated object configs")
    scale = np.asarray(body["scale"], dtype=float)
    info = json.loads(Path(body["info_path"]).read_text())
    if np.allclose(scale, scale[0], rtol=1e-8, atol=0):
        diagonal = float(np.linalg.norm(info["obb"])*scale[0])
        size_source = "scaled OBB diagonal from object metadata"
    else:
        import trimesh
        vertices = np.asarray(trimesh.load(body["file_path"], force="mesh").vertices)*scale
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        size_source = "scaled mesh AABB diagonal for nonuniform scales"
    center = (np.asarray(info["gravity_center"])*scale) @ quat2mat(body["pose"][3:]).T + body["pose"][:3]
    poses, labels, locations, input_counts = [], [], [], {}
    digest = hashlib.sha256()
    for taxonomy_index, taxonomy in enumerate(taxonomies):
        path = library / "succ_collect" / taxonomy / f"{relative}.npy"
        sample = np.load(path, allow_pickle=True).item()
        qpos = np.asarray(sample[f"{phase}_qpos"], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 29:
            raise ValueError(f"Invalid hand pose array: {path}")
        if sample["scene_path"] != str(scene_path) or sample["target_name"] != name:
            raise ValueError(f"Wrong object-scale identity: {path}")
        effective = np.asarray(sample["effective_object_scale"])
        if len(qpos) and not np.allclose(effective, scale, rtol=1e-6):
            raise ValueError(f"Mixed physical scales: {path}")
        count = len(qpos)
        input_counts[taxonomy] = count
        digest.update(taxonomy.encode())
        digest.update(np.ascontiguousarray(qpos).tobytes())
        poses.append(qpos)
        labels.extend([taxonomy]*count)
        locations.append(np.column_stack((np.full(count, taxonomy_index, np.int32),
                                          np.arange(count, dtype=np.int32))))
    return {"qpos": np.concatenate(poses), "labels": np.asarray(labels),
            "locations": np.concatenate(locations), "object": name, "scale": scale.tolist(),
            "diagonal": diagonal, "center": center, "size_source": size_source,
            "input_counts": input_counts, "qpos_sha256": digest.hexdigest()}


def analyze_group(job):
    index, library, run, relative, taxonomies, phase = job
    try:
        data = load_group(Path(library), relative, taxonomies, phase)
        metrics, selected, _ = analyze_grasps(data["qpos"], data["labels"], data["diagonal"], data["center"])
        sampling = Path(run) / "sampling" / relative / "epoch_000.npz"
        seed_info = None
        if sampling.exists():
            with np.load(sampling, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                points, inplane = metadata["point_count"], metadata["inplane_count"]
                frames = archive["frames_world"].reshape(inplane, points, 3, 3)
                unique = []
                for frame in frames:
                    if not any(np.allclose(frame, previous, atol=1e-6, rtol=1e-6) for previous in unique):
                        unique.append(frame)
                seed_info = {"surface_points": points, "roll_entries": inplane,
                             "unique_roll_directions": len(unique)}
        metrics.update(object_id=data["object"], relative=relative, scale=data["scale"],
                       object_size_source=data["size_source"], qpos_sha256=data["qpos_sha256"],
                       input_taxonomy_counts=data["input_counts"], seed_sampling=seed_info,
                       group_key=f"g_{index:06d}")
        return metrics, data["locations"][selected]
    except Exception as exc:
        return {"relative": relative, "group_key": f"g_{index:06d}",
                "error": f"{type(exc).__name__}: {exc}"}, None


CSV_COLUMNS = (
    "group_key", "object_id", "relative", "scale", "input_grasps", "finite_grasps", "invalid_grasps",
    "representative_count", "representative_fraction", "greedy_redundant_fraction",
    "strict_neighbor_fraction", "default_neighbor_fraction", "relaxed_neighbor_fraction",
    "default_near_pair_count", "numerically_identical_pair_count", "object_diagonal_m",
    "valid_taxonomy_count", "effective_taxonomy_count", "representative_taxonomy_count",
    "occupied_direction_bins", "effective_direction_count", "nearest_position_median_mm",
    "global_pair_median_position_mm", "global_pair_median_rotation_deg", "global_pair_median_joint_rms_deg",
    "unique_roll_directions", "roll_entries", "assessment", "review_flags", "qpos_sha256",
    "input_taxonomy_counts",
)


def csv_row(metrics):
    row = {column: metrics.get(column) for column in CSV_COLUMNS}
    row.update(scale=json.dumps(metrics["scale"]), review_flags=json.dumps(metrics["review_flags"]),
               input_taxonomy_counts=json.dumps(metrics["input_taxonomy_counts"]))
    for name in TOLERANCES:
        row[f"{name}_neighbor_fraction"] = metrics["near_duplicates"][name]["fraction_with_neighbor"]
    row.update(default_near_pair_count=metrics["near_duplicates"]["default"]["pair_count"],
               valid_taxonomy_count=metrics["taxonomy_diversity"]["occupied"],
               effective_taxonomy_count=metrics["taxonomy_diversity"]["effective_count"],
               representative_taxonomy_count=len(metrics["representative_taxonomy_counts"]),
               occupied_direction_bins=metrics["spatial_direction_diversity"]["occupied"],
               effective_direction_count=metrics["spatial_direction_diversity"]["effective_count"])
    if metrics["nearest_position_fraction_quantiles"]:
        row["nearest_position_median_mm"] = metrics["nearest_position_fraction_quantiles"]["median"]*metrics["object_diagonal_m"]*1000
    if metrics["global_pairwise"]["pair_count"]:
        row.update(global_pair_median_position_mm=metrics["global_pairwise"]["position_fraction"]["median"]*metrics["object_diagonal_m"]*1000,
                   global_pair_median_rotation_deg=metrics["global_pairwise"]["rotation_deg"]["median"],
                   global_pair_median_joint_rms_deg=metrics["global_pairwise"]["joint_rms_deg"]["median"])
    if metrics["seed_sampling"]:
        row.update({key:metrics["seed_sampling"][key] for key in ("unique_roll_directions", "roll_entries")})
    return row


def write_npy_member(archive, name, array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    archive.writestr(f"{name}.npy", stream.getvalue())


def scale_summary(metrics):
    groups = defaultdict(list)
    for item in metrics:
        groups[tuple(item["scale"])].append(item)
    result = []
    for scale, items in sorted(groups.items()):
        total = sum(item["finite_grasps"] for item in items)
        representatives = sum(item["representative_count"] for item in items)
        result.append({"scale": list(scale), "object_scales": len(items), "finite_grasps": total,
                       "representative_count": representatives,
                       "greedy_redundancy": 1-representatives/total if total else 0.,
                       "needs_review": sum(item["assessment"] == "needs_review" for item in items)})
    return result


def write_review_reports(output, summary, metrics):
    review = sorted((item for item in metrics if item["assessment"] == "needs_review"),
                    key=lambda item: (item["representative_count"], -item["greedy_redundant_fraction"]))
    with (output / "needs_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_row(item) for item in review)
    lines = [
        f"本次按单物体、单尺度分别分析 {summary['analyzed_object_scales']:,} 个已完成组合，"
        f"覆盖 {summary['distinct_objects']:,} 个物体。各组合均完成当前参数指定的全部抓取类型和轮数。",
        "",
        f"共 {summary['finite_grasps']:,} 个有限有效抓取；默认阈值下选出 "
        f"{summary['separated_representatives']:,} 个彼此分离的代表抓取，"
        f"可合并冗余约 {100*summary['weighted_greedy_redundancy']:.2f}%。",
        "",
        f"{summary['assessment_counts'].get('passes_heuristic', 0):,} 个组合通过默认筛查，"
        f"{summary['assessment_counts'].get('needs_review', 0):,} 个组合需要复查。"
        "这些标记是工程筛查结果，物体形状与可抓面会影响结论。",
        "",
        "近重复需要同时满足：手根位置距离 ≤ 物体实际 OBB 对角线的 2%；"
        "旋转差 ≤ 10°；22 个手指关节的 RMS 角差 ≤ 5°；任一关节角差 ≤ 15°。"
        "四元数 q 与 -q 按同一旋转处理。",
        "",
        f"默认阈值下，{100*summary['weighted_near_neighbor_fractions']['default']:.2f}% 的抓取存在近似邻居。"
        "该比例包含相似对的两端，代表抓取筛选后的可合并冗余比例更低。",
        "",
        "| 尺度 | 已完成组合 | 有效抓取 | 代表抓取 | 冗余 | 需复查 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["scale_summary"]:
        scale = item["scale"][0] if len(set(item["scale"])) == 1 else item["scale"]
        lines.append(f"| {scale} | {item['object_scales']:,} | {item['finite_grasps']:,} | "
                     f"{item['representative_count']:,} | {100*item['greedy_redundancy']:.2f}% | {item['needs_review']:,} |")
    lines += ["", "各尺度统计包含的物体集合不同。上述表格不能单独证明尺度变化造成了指标变化。", "",
              "复查条件为：代表抓取不足 64 个、可合并冗余超过 30%、手根空间方向覆盖不足 48 个区间中的 8 个，或出现无效姿态。",
              "多个条件可以同时命中；无有效抓取的组合单独标记。", ""]
    descriptions = {"fewer_than_64_separated_representatives": "代表抓取不足 64 个",
                    "more_than_30_percent_greedy_redundancy": "冗余超过 30%",
                    "fewer_than_8_of_48_root_position_directions": "空间方向覆盖不足",
                    "no_valid_grasps": "没有有效抓取", "invalid_qpos": "存在无效姿态"}
    for flag, count in summary["review_flag_counts"].items():
        lines.append(f"- {descriptions[flag]}：{count:,} 个组合。")
    lines += ["", f"{summary['seed_groups_with_repeated_roll_direction']:,} 个组合的原始采样包含重复滚转方向。"
              "当前 8 个角度条目首尾方向重合，实际为 7 个独立方向；最终抓取的重复情况按上述联合指标实测。",
              "", "原始抓取库保持完整。代表抓取索引保存为独立视图：选中的抓取两两不满足默认近重复条件，"
              "每个未选中的有限抓取都与某个选中抓取直接接近。该性质不证明已覆盖全部物理可行抓取。",
              "", "[逐尺度指标](object_scale_metrics.csv) · [复查清单](needs_review.csv) · "
              "[详细示例](examples.json) · [统计图](diversity_report.png) · [代表抓取索引](representatives.npz)", ""]
    (output / "report.md").write_text("\n".join(lines))


def load_representative_grasps(report_directory, relative):
    """Load the separated view, refusing an index for a changed source snapshot."""
    report_directory = Path(report_directory)
    summary = json.loads((report_directory / "summary.json").read_text())
    with (report_directory / "object_scale_metrics.csv").open(newline="") as stream:
        record = next((row for row in csv.DictReader(stream) if row["relative"] == relative), None)
    if record is None:
        raise ValueError(f"Object-scale was not successfully analyzed: {relative}")
    with np.load(report_directory / "representatives.npz", allow_pickle=False) as archive:
        taxonomies = archive["taxonomy_labels"].tolist()
        locations = archive[record["group_key"]].copy()
    data = load_group(Path(summary["source_library"]), relative, taxonomies, summary["phase"])
    if data["qpos_sha256"] != record["qpos_sha256"]:
        raise ValueError("Source snapshot changed; rerun diversity analysis before using this index")
    offsets = np.cumsum([0]+[data["input_counts"][name] for name in taxonomies])
    selected = offsets[locations[:, 0]]+locations[:, 1]
    return {"qpos": data["qpos"][selected], "taxonomies": data["labels"][selected],
            "source_file_rows": locations, "relative": relative, "phase": summary["phase"]}


def render_plot(output, metrics, example_job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    usable = [item for item in metrics if item["finite_grasps"]]
    axes[0, 0].hist([100*item["greedy_redundant_fraction"] for item in usable], bins=30, color="#2874bd")
    axes[0, 0].axvline(30, color="#c2410c", linestyle="--", label="Review threshold: 30%")
    axes[0, 0].set(xlabel="Greedy redundancy (%)", ylabel="Object-scale groups", title="A  Pose redundancy at default tolerances")
    axes[0, 0].legend()
    axes[0, 1].scatter([item["finite_grasps"] for item in usable],
                        [item["representative_count"] for item in usable], s=9, alpha=.4, color="#2874bd")
    axes[0, 1].set(xlabel="Raw successful grasps", ylabel="Separated representatives", title="B  Independent poses per object-scale")
    axes[1, 0].hist([item["spatial_direction_diversity"]["occupied"] for item in usable], bins=np.arange(50)-.5, color="#2874bd")
    axes[1, 0].set(xlabel="Occupied root-position direction bins (of 48)", ylabel="Object-scale groups", title="C  Spatial coverage")
    _, library, _, relative, taxonomies, phase = example_job
    sample = load_group(Path(library), relative, taxonomies, phase)
    example = next(item for item in metrics if item["relative"] == relative)
    with np.load(output / "representatives.npz", allow_pickle=False) as index:
        locations = index[example["group_key"]]
    offsets = np.cumsum([0]+[sample["input_counts"][name] for name in taxonomies])
    selected = offsets[locations[:, 0]]+locations[:, 1]
    position = sample["qpos"][:, :3]*1000
    axes[1, 1].scatter(position[:, 0], position[:, 1], s=8, color="#9ca3af", alpha=.5, label="All grasps")
    axes[1, 1].scatter(position[selected, 0], position[selected, 1], s=5, color="#c2410c", alpha=.5, label="Separated representatives")
    axes[1, 1].set(xlabel="Root X (mm)", ylabel="Root Y (mm)", title=f"D  Example: {example['object_id'][:8]}, scale {example['scale'][0]:g}")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].legend()
    figure.savefig(output / "diversity_report.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="output/single_object_grasps_dgn_shadow")
    parser.add_argument("--output", default="output/grasp_diversity_dgn_shadow")
    parser.add_argument("--object", default="*")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--phase", choices=("pregrasp", "grasp", "squeeze"), default="grasp")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 audits every completed object-scale")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    library, output = resolve_path(args.library), resolve_path(args.output)
    if library == output or library in output.parents or output in library.parents:
        parser.error("Analysis output must be separate from the source library")
    runtime = json.loads((library / "runtime.json").read_text())
    run = Path(runtime.get("run_directory", library))
    relatives, settings = completed_groups(run, args.object)
    if args.scale is not None:
        relatives = [relative for relative in relatives if np.allclose(
            np.load(library / "scene_cfg" / f"{relative}.npy", allow_pickle=True).item()["scene"][relative.split("/")[0]]["scale"], args.scale)]
    if args.limit:
        relatives = relatives[:args.limit]
    if not relatives:
        raise ValueError("No fully completed object-scale groups match this selection")
    jobs = [(index, str(library), str(run), relative, settings["taxonomies"], args.phase)
            for index, relative in enumerate(relatives)]
    started = datetime.now(timezone.utc).isoformat()
    metrics, errors = [], []
    with output_lock(output):
        atomic_json(output / "progress.json", {"state": "running", "planned_groups": len(jobs),
                    "completed_groups": 0, "started_at": started})
        with (output / "object_scale_metrics.csv").open("w", newline="") as csv_file, \
                zipfile.ZipFile(output / "representatives.npz.tmp", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive, \
                ProcessPoolExecutor(max_workers=args.workers) as pool:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            write_npy_member(archive, "taxonomy_labels", np.asarray(settings["taxonomies"]))
            write_npy_member(archive, "group_relatives", np.asarray(relatives))
            for completed, (result, selected) in enumerate(pool.map(analyze_group, jobs, chunksize=1), 1):
                if "error" in result:
                    errors.append(result)
                else:
                    metrics.append(result)
                    writer.writerow(csv_row(result))
                    write_npy_member(archive, result["group_key"], selected.astype(np.int32))
                if completed % 25 == 0 or completed == len(jobs):
                    csv_file.flush()
                    atomic_json(output / "progress.json", {"state": "running", "planned_groups": len(jobs),
                        "completed_groups": completed, "errors": len(errors), "started_at": started})
                    print(f"Analyzed {completed}/{len(jobs)} completed object-scale groups; errors={len(errors)}", flush=True)
        (output / "representatives.npz.tmp").replace(output / "representatives.npz")
        total = sum(item["finite_grasps"] for item in metrics)
        representatives = sum(item["representative_count"] for item in metrics)
        summary = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                   "source_library": str(library), "source_checkpoint_run": str(run), "phase": args.phase,
                   "source_sampling_points": settings["points"], "source_inplane_entries": settings["inplane"],
                   "selected_completed_object_scales": len(jobs), "analyzed_object_scales": len(metrics),
                   "distinct_objects": len({item["object_id"] for item in metrics}), "errors": errors,
                   "finite_grasps": total, "separated_representatives": representatives,
                   "weighted_representative_fraction": representatives/total if total else 0.,
                   "weighted_greedy_redundancy": 1-representatives/total if total else 0.,
                   "weighted_near_neighbor_fractions": {name: sum(item["near_duplicates"][name]["grasps_with_neighbor"] for item in metrics)/total if total else 0. for name in TOLERANCES},
                   "group_greedy_redundancy_quantiles": dict(zip(("p05", "median", "p95"), map(float,np.quantile([item["greedy_redundant_fraction"] for item in metrics], [.05,.5,.95])))) if metrics else None,
                   "assessment_counts": dict(Counter(item["assessment"] for item in metrics)),
                   "review_flag_counts": dict(Counter(flag for item in metrics for flag in item["review_flags"])),
                   "scale_summary": scale_summary(metrics),
                   "seed_groups_with_repeated_roll_direction": sum(item["seed_sampling"] is not None and item["seed_sampling"]["unique_roll_directions"] < item["seed_sampling"]["roll_entries"] for item in metrics),
                   "method": {"near_duplicate": {name: vars(value) for name,value in TOLERANCES.items()},
                              "position_units": "fraction of physical object OBB diagonal (AABB for nonuniform scales)",
                              "rotation": "SO(3) geodesic angle; normalized quaternion q and -q are equivalent",
                              "joints": "RMS and maximum absolute difference across all 22 bounded finger joints, in degrees",
                              "local_duplicate_counts": "exact within the stated position/rotation/joint tolerances",
                              "global_pair_quantiles": "all pairs for <=20,000 pairs, otherwise 20,000 deterministic uniform random pairs",
                              "representatives": "deterministic maximal independent set; direct coverage, no transitive merging",
                              "spatial_coverage": "root position direction from object center, 48 equal-area Fibonacci bins; not contact-area coverage",
                              "review_thresholds": "<64 representatives, >30% greedy redundancy, <8 occupied direction bins, or invalid qpos",
                              "assessment_limits": "Engineering screening thresholds, not proof of feasible grasp-manifold coverage; shape constraints can explain flags",
                              "representative_index_format": "g_XXXXXX rows are [taxonomy_labels index, row in published source file]; CSV qpos_sha256 identifies the analyzed snapshot",
                              "source_library_modified_by_analysis": False}}
        atomic_json(output / "summary.json", summary)
        write_review_reports(output, summary, metrics)
        examples = sorted(metrics, key=lambda item: item["relative"])[:4]
        atomic_json(output / "examples.json", examples)
        if metrics:
            example = examples[0]
            render_plot(output, metrics, jobs[int(example["group_key"].split("_")[1])])
        atomic_json(output / "progress.json", {"state": "complete", "planned_groups": len(jobs),
            "completed_groups": len(jobs), "errors": len(errors), "started_at": started,
            "finished_at": summary["finished_at"]})
        print(json.dumps({key:summary[key] for key in ("analyzed_object_scales", "distinct_objects", "finite_grasps",
               "separated_representatives", "weighted_greedy_redundancy", "assessment_counts", "review_flag_counts")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
