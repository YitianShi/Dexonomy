# DGN 与 GraspNet 单物体抓取生成

当前任务只处理独立物体：资产准备 → 表面采样与模板匹配 → MuJoCo 优化 → 物理验证 → 保存抓取库。
不需要已有杂乱场景。物体位于原点，环境中没有桌面或其他物体。

## 数据与保存位置

| 数据集 | 物体配置 | 尺度 | 新生成抓取库 |
|---|---|---|---|
| DGN | `assets/object/DGN_5k/scene_cfg` | 资产中的 6 个原有尺度 | `output/single_object_grasps_dgn_shadow` |
| GraspNet-1Billion | `assets/object/graspnet_1billion/scene_cfg` | 原始米制坐标，`[1,1,1]` | `output/single_object_grasps_graspnet_shadow` |

DGN 的 5,697 个物体有 34,182 个物体/尺度组合。
GraspNet 采用官方 `000–087` 全部 88 个物体，每个物体使用原始实际尺寸，
不套用 DGN 的归一化尺度。
官方 Dexonomy 抓取仍保存在 `data/extracted/Dexonomy_GRASP_shadow/succ_collect`。
新生成结果与官方库分别保存。

## 准备 GraspNet 资产

在 VS Code 选择 **GraspNet：下载并接入全部 88 个物体**，或运行：

```bash
python scripts/prepare_graspnet_objects.py --download
```

来源是 [DexGraspNet 2.0 官方模型包](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0/tree/main)
中的 `meshdata.tar.gz`。导入脚本校验官方 SHA256，提取 `coacd` 碰撞文件和
`simplified.obj` 表面网格，保留 URDF 中的固定连接、网格尺度与局部坐标。
它复用已有凸分解，生成 MuJoCo XML、质量与重心信息、独立物体配置；不重新运行 CoACD。
低于 MuJoCo 体积计算阈值的微小凸块仍参与碰撞，但不参与质量计算；
对应网格和处理记录保存在导入清单中，物体尺寸不变。
训练图片、点云缓存和杂乱场景文件不参与当前单物体流程。

压缩包缓存到 `downloads/graspnet_1billion_meshdata.tar.gz`；该目录可写，
已有官方数据所在的 `data` 挂载保持只读。
导入后通过 `assets/object/graspnet_1billion/import_manifest.json` 核对
88 个编号、各物体的凸块数量、实际尺寸和资产路径。

## 生成与验证

两个数据集各有独立的 Python 入口和 VS Code 启动项：

| 数据集 | Python 入口 | VS Code 启动项 | 默认物体 |
|---|---|---|---|
| GraspNet | `scripts/generate_graspnet_single_object_grasps.py` | **1. GraspNet：单物体抓取生成与验证** | `000` |
| DexGraspNet | `scripts/generate_dexgraspnet_single_object_grasps.py` | **2. DexGraspNet：单物体抓取生成与验证** | `core_bottle_1071fa4cddb2da2fc8724d5673a063a6` |

入口固定各自的资产和输出路径，只需选择物体，不需要配置场景。
GraspNet 输入 `000–087` 的编号，DexGraspNet 使用完整物体名称。
输入 `*` 可遍历该数据集的全部物体；每个物体/尺度仍在独立环境中生成与验证。
默认 GraspNet 处理一个实际尺寸，DexGraspNet 处理所选物体的 6 个原有尺度。

命令行生成一个物体：

```bash
python scripts/generate_graspnet_single_object_grasps.py --object 000
python scripts/generate_dexgraspnet_single_object_grasps.py \
  --object core_bottle_1071fa4cddb2da2fc8724d5673a063a6
```

两个入口复用同一套原生 Dexonomy 代码，参数相同：

- 33 种 Shadow 手抓取模板，合计 275 个模板接触点。
- 默认 `--all-hand-contacts`：每个手部接触点依次作为初始化锚点，
  分别遍历相同的 128 个物体采样点 × 8 个旋转角度条目。
- 1 轮；每个手部接触点的 1,024 条候选均进入优化与验证。
  每类模板共验证 `该模板接触点数 × 1,024` 条，不做候选数量截断。
- GPU 2 执行模板匹配；84 个 CPU 工作进程执行 MuJoCo 优化与物理验证。
- `skip_traj=true` 跳过机械臂轨迹规划；手部接近、闭合与物理验证仍执行。
- 只保存成功候选的详细数组，保留所有候选的尝试数和失败原因计数。

先预览 GraspNet 任务规模：

```bash
python scripts/generate_graspnet_single_object_grasps.py --object '*' --dry-run
python scripts/generate_dexgraspnet_single_object_grasps.py --object '*' --dry-run
```

全量 GraspNet 包含 88 × 275 = 24,200 个接触点工作组、24,780,800 条待验证候选。
候选数量不是最终有效抓取数量。
移除 `--dry-run` 开始生成；选择单个物体或添加 `--taxonomies 24_Tip_Pinch` 可缩小范围。
两个入口共享 `generate_single_object_grasps.py` 中的生成实现，避免复制优化和验证逻辑。

## 进度与复用

实际抓取位于各库的 `succ_collect/<taxonomy>/<object>/floating/scale_<hash>.npy`。
每条记录含预抓取、抓取、挤压姿态以及候选来源。

```bash
watch -n 2 cat output/single_object_grasps_graspnet_shadow/progress.json
```

`completed_groups/planned_groups` 是本次接触点工作组进度；`attempted` 是已验证候选数；
`new_valid_grasps` 是本次成功数；`library_valid_grasps` 包括同一库历史成功结果。
`runtime.json` 记录进程、参数和状态。再次运行相同配置会跳过已完成组。
改变采样参数会建立独立检查点，成功结果仍合并进该数据集的固定抓取库。

遍历全部接触点的检查点位于
`runs/all_contacts_points_128_inplane_8_epochs_1`，旧随机锚点生成的数组与检查点保留。
每个物体/尺度/类型下按 `contact_000`、`contact_001` 等保存独立批次。
每条候选记录 `hand_contact_indices`、`sample_point_indices`、`inplane_indices`；
已发布抓取中的 `hand_contact_indices=-1` 表示旧随机模式没有记录锚点编号。
每个接触点对应整个模板参与联合优化，其他接触点也保留在匹配目标中。
