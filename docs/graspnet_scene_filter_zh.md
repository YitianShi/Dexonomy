# 官方 GraspNet 场景中的 MuJoCo 抓取过滤

这是独立的第二步：复用第一步的单物体抓取库，在官方场景中验证已有抓取。
不重新生成或优化抓取，不重新投放、缩放或沉降原布局中的物体；零抓取结果也保留原场景。

## 新增 450 个 GraspNet 测试场景

2026-11-12 已将官方 Dense 90、Loose 180、Random 180 个场景转换为 1,769 个目标配置，并完成全部过滤，保留 2,108,984 条抓取。444 个场景有保留抓取，6 个场景过滤后为零，任务错误为 0；详见[完成核对记录](../output/graspnet_test_scene_filter_shadow/completion_verification.json)。配置与训练场景使用不同的场景标识和目录，已有单物体库及训练场景结果保留。

为与此前结果保持一致，测试场景过滤使用 MuJoCo 3.3.3 回退环境。VS Code 入口为“6. GraspNet：450 个官方测试场景抓取过滤”。独立脚本固定该数据集的输入与输出路径，以下命令也可在中断后续跑：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .envs/mujoco333/bin/python scripts/filter_graspnet_test_scene_grasps.py \
  --workers 84 --approach-steps 8 --check-pregrasp --no-lift
```

输出与实时进度位于 [output/graspnet_test_scene_filter_shadow](../output/graspnet_test_scene_filter_shadow/)，启动命令和 Python 环境记录在 `launch.json`。任务在后台运行时不需重复启动，运行锁会阻止相同输出目录的第二个任务。查看进度：

```bash
watch -n 2 cat output/graspnet_test_scene_filter_shadow/progress.json
```

## 启动

VS Code 选择 **4. GraspNet：官方场景 MuJoCo 抓取过滤**，按 F5。
默认输入 `scene_0000`；输入 `scene_*` 选择全部 7,600 个已导入场景、37,008 个目标配置。

```bash
python scripts/filter_graspnet_scene_grasps.py --scene scene_0000 --workers 84
# 全部训练场景：
python scripts/filter_graspnet_scene_grasps.py --scene 'scene_*' --workers 84
# 检查任务数量与参数，不运行过滤：
python scripts/filter_graspnet_scene_grasps.py --scene 'scene_*' --dry-run
# 小规模测试；每个目标/类型最多验证 64 条：
python scripts/filter_graspnet_scene_grasps.py --scene scene_0000 --target 014 \
  --taxonomies 24_Tip_Pinch --max-per-target 64 --workers 1
```

默认读取 `assets/scene/dexgraspnet2_train/scene_cfg` 和
`output/single_object_grasps_graspnet_shadow/succ_collect`，使用 Shadow 手。
每个目标遍历全部 33 类抓取，默认不截断候选、不做 FPS。
`--target`、`--taxonomies` 可选择目标和类型；`--max-per-target 0` 表示验证所有尺度兼容候选。

启动项使用 84 个 CPU 工作进程并行处理不同“场景/目标”，不使用 GPU，也不调用机械臂规划。
每个进程加载一次完整静态 MuJoCo 场景，依次验证各类型抓取，通过重置手姿态复用模型。
BLAS/OpenMP 每进程限制为一个线程；单独运行过滤时可使用 84 个工作进程。
同时运行两步时，可将过滤改为 `--workers 16`，与 84 进程的单物体生成合计最多 100 个工作进程。

## 变换与过滤规则

对每条抓取的 `pregrasp`、`grasp`、`squeeze`，将手腕从源物体坐标系映射到场景：

$$
T^{c}_{WH}=T^{c}_{WO}\,(T^{s}_{WO})^{-1}T^{s}_{WH},
\qquad q^{c}_{\mathrm{finger}}=q^{s}_{\mathrm{finger}}.
$$

这里 $s$ 为单物体库，$c$ 为场景，$W/O/H$ 为世界/物体/手腕。
GraspNet 原始米制尺度固定为 $[1,1,1]$。已有抓取的有效尺度为
$s_\mathrm{eff}=s_\mathrm{source}\odot s_\mathrm{scene\_scale}$，要求：

$$
\max_i\frac{|s_{\mathrm{eff},i}-s_{\mathrm{target},i}|}
{|s_{\mathrm{target},i}|}\leq 0.02.
$$

新入口默认检查精确 `pregrasp` 起点，随后检查 8 个插值样本（含 `grasp`）。
手腕位置线性插值、旋转使用已有四元数插值函数、手指关节线性插值。
所有这些姿态以及 `squeeze` 的手/桌面、手/邻物、自碰撞距离均须 $d\geq 0$；
接近路径对目标的穿透须 $d\geq -0.001\,\mathrm m$。
`squeeze` 是闭合控制目标，允许其对目标的静态穿透，但要求在 2 mm 接触裕量内检测到目标接触。
每条候选遇到第一个失败条件即停止，失败计数按首次失败原因归类。
离散检查不提供连续碰撞保证；可增加 `--approach-steps` 提高采样密度。
`--no-check-pregrasp` 可恢复旧 Hydra 入口跳过精确起点的行为。

加 `--lift` 会在几何通过后执行重力下的接近、闭合、抬升与保持：
所有刚性物体质量设为 0.1 kg，滑动摩擦系数 0.2，抬升控制目标 12 cm，
实际目标上升至少 5 cm 才保留。每条测试重置完整场景，不提前修改官方布局。
这里的动力学使用本项目参数。默认只执行几何与接触过滤，源单物体抓取已通过第一步的动力学验证。

## 输出、进度与续跑

```text
output/graspnet_scene_filter_shadow/
├── progress.json                        # 当前选择范围的实时进度
├── run.lock                             # 防止重复启动同一个输出目录
└── runs/<filter_id>/                     # 参数、代码和模型版本决定的 ID
    ├── settings.json
    ├── progress.json
    ├── progress.sqlite                  # 每个场景/目标的完成记录及逐类型计数
    ├── errors.jsonl                     # 有失败任务时保存原因
    └── filtered_grasps/<taxonomy>/scene_XXXX/<target>.npy
```

结果保留三个阶段的姿态以及 `source_grasp_paths`、`source_indices`、有效尺度和过滤计数，
可追溯到原库。输出先写临时目录，源输入稳定时才发布结果并记为完成。
同参数重跑跳过未变且输出完整的目标；源文件新增、删除或原子替换后重新验证受影响目标。
源库在任务期间改变的目标记为 `changed_input_targets`，不发布此次结果，下次启动重试。
删除结果文件也会使该目标重跑。改过滤参数或代码/模型版本会使用新结果目录。
进入重验前会清理该目标的旧过滤结果，避免源输入变化或任务失败时被查看器误用；原始抓取库不受影响。
Ctrl+C/SIGTERM 停止本入口自己的工作进程，已提交检查点可续用。

```bash
watch -n 2 cat output/graspnet_scene_filter_shadow/progress.json
```

| 字段 | 含义 |
| --- | --- |
| `finished_targets / targets` | 当前选择中已完成库快照过滤的目标数 / 总目标数 |
| `resumed_targets` | 完成目标中复用已有检查点的数量 |
| `active_targets` | 已提交、尚未返回的任务数，包含队列中的任务 |
| `taxonomy_statuses.saved / empty` | 有保留抓取 / 已处理但保留数为零的目标与类型组合数 |
| `taxonomy_statuses.missing_source_grasp` | 源文件尚不存在的目标与类型组合数，不能解读为无可行抓取 |
| `counts.input_count` | 所有选中源文件的输入候选数 |
| `counts.evaluated_count` | 经过尺度筛选和可选调试上限后，实际进入验证的候选数 |
| `counts.valid_count` | 保留的抓取数 |
| `counts.rejected_*` | 尺度、碰撞、目标穿透、无目标接触、抬升失败计数 |
| `changed_input_targets / failed_targets` | 因源库变化而延期 / 发生错误的目标数 |
| `filtered_grasps` | 本轮实际结果根目录，可用于查看器 |

`completed_snapshot` 只表示本次读取的库版本已处理。第一步生成尚在运行时，
应待其完成后再次启动，以补齐新增抓取和之前缺失的源文件。
`incomplete` 表示源输入变化或任务失败；这些目标没有有效完成检查点。
调试上限生效时，输入数可能大于实际验证数加尺度拒绝数。

把进度中的 `filtered_grasps` 路径传给已有查看器即可显示完整场景及筛选结果：

```bash
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py \
  --root output/graspnet_scene_filter_shadow/runs/<filter_id>/filtered_grasps \
  --scene scene_0000
```

`<filter_id>` 要替换为实际目录名。查看器的 FPS 只决定显示数量，不修改抓取结果。
