# 当前最终抓取数据

更新日期：2026-11-12。

五套最终数据共 **55,205,354 条**，其中单物体抓取 22,908,743 条、场景实例 32,296,611 条。场景实例包含同一抓取在不同场景的复用，不代表同样数量的独立物体抓取。

数据保存在实际目录中，没有符号链接。每个 NPY 保留 `pregrasp_qpos`、`grasp_qpos`、`squeeze_qpos` 三阶段姿态、目标物体、当前 taxonomy、接触掩码及来源标识。单物体按 `<taxonomy>/<object>/floating/scale_*.npy` 组织，场景按 `<taxonomy>/<scene>/<target>.npy` 组织。

| 数据 | 实际路径 | 抓取条数 |
|---|---|---:|
| GraspNet 单物体（88 个） | `datasets/single_object/graspnet` | 2,550,154 |
| ShapeNet 单物体（937 个） | `datasets/single_object/shapenet` | 20,358,589 |
| GraspNet 训练场景 | `datasets/scenes/graspnet_train` | 30,555,699 |
| GraspNet 测试场景 | `datasets/scenes/graspnet_test` | 1,413,830 |
| ShapeNet 测试场景 | `datasets/scenes/shapenet_test` | 327,082 |

```text
output/
├── datasets/
│   ├── single_object/
│   │   ├── graspnet/
│   │   └── shapenet/
│   ├── scenes/
│   │   ├── graspnet_train/
│   │   ├── graspnet_test/
│   │   └── shapenet_test/
│   ├── scene_cfg/single_object/  # 1,025 份必要场景配置
│   ├── index.json
│   └── README.md
├── dataset_index.json
└── README.md
```

单物体数据需要的 1,025 份场景配置已经复制到 `datasets/scene_cfg/single_object/`，NPY 中的 `scene_path` 已同步更新。三套场景库继续使用 `assets/scene/` 中的官方场景布局；物体与手部模型继续使用 `assets/` 和 `third_party/`。这些输入资产未删除。

`source_grasp_paths`、`source_indices`、`dedup_source_path`、`dedup_source_indices` 仍保留历史来源文件／行号，作为追溯标识；对应的旧中间文件已删除，加载当前抓取姿态和场景不需要这些来源文件。旧批次的候选、去重前副本和断点数据库不再保留，旧批次不能续跑。

必要的接触审计、去重映射和参数已移到 `reports/cleanup/20261112_dataset_only/history/`；193 条此前剔除记录的可视化样例仍在 `visualizations/current_rejected_grasps_20261112/`，2 条待定样例保存于 `reports/current_rejected_grasps_20261112/ambiguous_samples/`。官方原始抓取 `data/extracted/` 未删除。

## 验证范围

全部 537,953 个抓取文件已逐个加载并核对条数、三阶段维度和有限值；共核对 36,388 个不同场景配置。33,263 个单物体抓取文件仅更新场景配置路径，其三阶段数组和其余所有字段保存前后逐项一致。其余抓取文件的 inode／大小／mtime 与生成时一致。旧目录不可访问时，五套查看器模型检查全部通过。

当前单物体覆盖 GraspNet 88 个物体、DexGraspNet 2.0 ShapeNet 937 个物体，均为本次生成使用的原始尺度。场景过滤处理了 GraspNet 训练 7,600、测试 450、ShapeNet 测试 219 个场景；无成功抓取的场景不产生非空文件。场景记录经过碰撞、接近路径与目标接触过滤，未做场景抬升验证。

接触复验允许额外目标接触并保持当前已分配类别；去重在同物体／尺度／类别内比较三阶段姿态，并保留不同实际接触部位组合。完整公式见 [单物体生成说明](single_object_grasps_zh.md)、[接触复验说明](taxonomy_contact_filter_zh.md)和[去重说明](grasp_dedup_zh.md)。

## 使用与后续生成

```bash
python scripts/view_clutter_grasps_mujoco.py --gui
python scripts/view_clutter_grasps_mujoco.py \
  --dataset deduplicated_graspnet_single --object 031 --max-grasps 20 --check
```

`output/dataset_index.json` 只登记五套最终结果。后续生成、过滤和去重的中间文件默认写入 `work_runs/`；工作目录中的新结果不会自动覆盖当前发布库。需要重验或再次去重时，脚本可直接发现当前五套数据，不要求已经删除的中间版本存在。

原始生成数量、多样性分析、历史运行配置和此前清理收据仍属于历史快照，不代表旧副本仍存在。此次删除与验证见 `reports/cleanup/20261112_dataset_only/summary.json`。
