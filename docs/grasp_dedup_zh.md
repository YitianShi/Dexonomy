# 已生成抓取去重

入口为 [deduplicate_grasps.py](../scripts/deduplicate_grasps.py)，核心实现在
[grasp_dedup.py](../dexonomy/util/grasp_dedup.py)。输入优先使用接触复验库 `taxonomy_filtered_*`；
精简目录后可直接读取当前五套 `deduplicated_*` 发布库。新工作结果单独写入 `work_runs/`，默认不替换当前发布数据。

## 判定范围

- 单物体：同一个物体、同一个尺度、同一个当前 taxonomy 标签内比较。
- 场景：同一个场景、同一个目标物体、同一个当前 taxonomy 标签内比较。
- 实际目标接触的手部部位掩码必须相同；不同接触组合分别保留。
- 同时检查 pregrasp、grasp、squeeze 三阶段；仅显示出来相似、但接近方式或收紧姿态不同的抓取仍可能保留。

不同场景中复用同一个物体抓取属于不同场景实例，不互相删除。场景内独立选代表，
因此不会因单物体库保留的某个代表在当前场景不可用而误删其他可用抓取。当前分类和此前改标保持不变，
不把不同 taxonomy 合并。此次不重新执行 MuJoCo 接触检查或动力学验证。

## 距离和阈值

令阶段 $s\in\{\mathrm{pregrasp},\mathrm{grasp},\mathrm{squeeze}\}$。
保存的姿态为自由根位置 $t_s$、归一化后的 wxyz 四元数 $q_s$ 和 22 个关节角 $\theta_s$。
本地 Shadow 模型从自由根到手掌的固定偏移为 $b=(0,0,0.034)^\top\,\mathrm m$，
因此手掌位置为 $p_s=t_s+R(q_s)b$。

对两条抓取 $i,j$，计算：

$$
d_p=\max_s\|p_{i,s}-p_{j,s}\|_2,
\qquad
d_R=\max_s 2\arccos\bigl(\operatorname{clip}(|q_{i,s}^\top q_{j,s}|,0,1)\bigr),
$$

$$
d_{\mathrm{rms}}=\max_s\sqrt{\frac1{22}\sum_{k=1}^{22}
  (\theta_{i,s,k}-\theta_{j,s,k})^2},
\qquad
d_{\max}=\max_{s,k}|\theta_{i,s,k}-\theta_{j,s,k}|.
$$

默认只有同时满足下列四个条件，且接触部位掩码、尺度一致，才判为可合并：

| 条件 | 阈值 |
|---|---:|
| 三阶段最大手掌位置差 $d_p$ | 2 mm |
| 三阶段最大旋转差 $d_R$ | 5° |
| 三阶段最大关节 RMS 差 $d_{\mathrm{rms}}$ | 3° |
| 任一阶段任一关节最大差 $d_{\max}$ | 8° |

这些是本次清理的保守工程阈值，不是论文统一标准，也不是 taxonomy 的接触距离阈值。
四元数 $q$ 与 $-q$ 被视为同一旋转；关节是有界关节，直接比较弧度差，不做周期取模。
因为一个比较组中的目标坐标变换固定，位置差和相对旋转角与转换到目标物体坐标系后相同。

## 保留代表的方法

按文件内原有顺序保留首个尚未被代表的抓取，并删除其阈值范围内的其他未处理抓取。
空间 KD-tree 只用来加速候选查找，最终仍逐项检查全部条件。
每条删除记录都直接指向一条保留记录。不会把“甲接近乙、乙接近丙、甲却远离丙”
的整条链合并；这种情况下仍保留丙。每个原本非空的物体／尺度／taxonomy 或场景／目标／taxonomy 文件
至少保留一条，且不会修改保留抓取的姿态或其验证标签。

输出的 `exact_pose_duplicates` 表示三阶段原始 29 维数组数值完全相同的删除记录；
`near_pose_duplicates` 是其余通过上述约束的删除记录。四元数符号相反的同一旋转也可以合并，
但若原始数组不同，会计入后者。比例以各库输入条数为分母，场景条数按场景实例计。

## 文件与来源

```text
work_runs/grasp_dedup_shadow/
├── progress.json
├── run.log
└── runs/<run_id>/
    ├── settings.json
    ├── progress.json
    └── <dataset_id>/
        ├── filtered_grasps/<taxonomy>/.../*.npy
        └── records/.../<object_scale_or_scene>.{npz,json}
```

`filtered_grasps` 是新任务的工作结果，实际只包含保留行。2026-11-12 整理后，当前发布的五套结果实际位于 `output/datasets/`；旧生成副本和断点已删除。
保留原有 `source_grasp_paths`、`source_indices`、接触掩码和分类记录，来源路径只作为历史追溯标识，不是当前数据的加载依赖。
发生删行的文件新增 `dedup_source_path`、`dedup_source_indices`，指向去重前复验文件的行。
未发生删行的文件按字节复制。官方原始抓取不参与处理。

每个 NPZ 包含 `source_files`、`file_offsets` 和 `representative_indices`。
用 `file_offsets[f]:file_offsets[f+1]` 取出第 $f$ 个文件的映射，映射的值是该**输入文件**中的代表行号。
值等于自身行号的是保留行，其余是删除行。查找输出行号时，在递增的保留行号列表中查找该代表。
同目录 JSON 保存输入／输出文件状态、逐组计数、最大删除距离和示例，支持断点检查。

程序对每条删除记录再次检查其与保留代表的四项距离；所有改写文件保存后重新读取，
逐字段核对为输入的对应子集，包括姿态、已改标标签和来源索引。完整运行还核对输入条数和文件数与目录索引一致，
没有错误才发布五个去重库。

## 使用

```bash
# 读取当前五套库，32 个 CPU 进程；工作结果写入 work_runs，不覆盖发布库
OPENBLAS_NUM_THREADS=1 python scripts/deduplicate_grasps.py --workers 32

# 查看选择范围，不写文件
python scripts/deduplicate_grasps.py --dry-run

# 仅检查物体 031，独立输出；限定选择不注册为正式库
python scripts/deduplicate_grasps.py --datasets graspnet_single \
  --group '031/*' --output /tmp/grasp_dedup_031 --workers 1
```

当前发布库在通用查看器中的名称包含“接触复验＋去重”。例如：

```bash
python scripts/view_clutter_grasps_mujoco.py \
  --dataset deduplicated_graspnet_single --object 031 --max-grasps 20
```

`--position-mm`、`--rotation-deg`、`--joint-rms-deg`、`--joint-max-deg` 可调整阈值。
参数和实现签名决定运行目录；相同输入和实现支持续跑。改变阈值会生成另一套结果，不覆盖此前去重库。

当前目录索引只保留五套最终库时，去重脚本会使用它们作为输入。新任务在 `work_runs/` 内按参数与实现签名分批保存，可续跑新任务；已删除的旧批次不再支持续跑。
