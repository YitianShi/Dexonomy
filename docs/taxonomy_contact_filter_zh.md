# 已生成抓取的接触复验与恢复

2026-11-12 数据精简后，当前五套发布库位于 `output/datasets/`。旧生成／复验副本和断点已删除，历史接触审计仅保存在 `reports/cleanup/20261112_dataset_only/history/`。下面的数量与恢复过程属于历史记录；重新复验当前数据会创建新的 `work_runs/` 任务，不能续跑旧批次。

当前默认允许额外目标接触：已分配类别的抓取（含已改标的抓取）全部保持当前标签；
只将此前被剔除或归入歧义区、且满足原类别全部必需接触组的抓取恢复到原类别。
缺少必需接触组的抓取不因这项放宽而恢复。额外接触部位和旧的严格判定仍保存在审计中。
`--strict-extra-contacts` 可显式运行原来的严格规则。

这一步读取已有的五套生成数据，重放 `grasp_qpos`，检查手与**目标物体**的接触部位。
预抓取本来可以没有接触，squeeze 是闭合控制目标，二者不用于此次类别判定。
物体固定在该数据记录的场景位姿，只查询接触，不重新生成、优化或步进仿真。
默认使用原生成环境 MuJoCo 3.3.3、84 个 CPU 进程，不使用 GPU。

## 判定规则

令实际接触的手部部件集合为 $C$；模板 $t$ 的 `hand_cbody` 集合为 $A_t$，
`required_cbody` 中的每个分组为 $R_{t,k}$。距离不大于 2 mm 的目标物体接触计入 $C$。

原来的严格模板接触规则为：

$$
C\ne\varnothing,\qquad C\subseteq A_t,\qquad
\forall k,\ C\cap R_{t,k}\ne\varnothing.
$$

对尚未分配类别的抓取，当前恢复条件为：

$$
C\ne\varnothing,\qquad \forall k,\ C\cap R_{t,k}\ne\varnothing.
$$

恢复时不再要求 $C\subseteq A_t$。已改标的记录直接保留现有类别，不转回原类别。

邻近物体、桌面与手的接触不会代替目标接触。此次检查保留既有碰撞与物理验证结果，
不重新声称通过受力或抬升验证。模型位置阶段查询已与原 `mj_forward` 接触结果对照。

严格规则的结果如下，当前默认在此基础上仅恢复满足上述条件的剔除／歧义记录：

| 结果 | 处理 |
|---|---|
| 原类别满足规则 | 保留原类别，即使还有其他类也符合 |
| 原类别不符，且唯一其他类别符合 | 改到该类别 |
| 原类别不符，但多个类别符合 | 保存到 `ambiguous_grasps`，记录全部候选，暂不硬分配类别 |
| 无目标接触或不符合任何类别 | 从新结果中剔除，逐条审计记录保留 |
| 非法姿态 | 剔除并单独计数 |

33 类模板中有 7 对具有完全相同的接触部位规则，例如 Tip Pinch 与 Palmar Pinch。
因此这种复验给出的是**接触部位一致性**；不能单凭接触部位证明完整的姿态 taxonomy 分类。

实际触点到原模板同部件标注点的最近距离，也作为诊断保存到
`source_template_max_point_distance_m`。此值不参与删选；`inf` 表示存在原模板未标注部件的接触，
`nan` 表示无接触或无合法姿态。模板离散点不会被当作整片允许接触区域的精确边界。

## 运行与进度

VS Code 启动项：**7. 全部已生成抓取：接触复验（允许额外接触，保留已改标）**。

```bash
.envs/mujoco333/bin/python scripts/refilter_taxonomy_contacts.py --workers 84
```

历史恢复曾使用以下命令复用严格审计；旧输入副本现已删除，此命令不再适用于当前目录：

```bash
.envs/mujoco333/bin/python scripts/refilter_taxonomy_contacts.py \
  --workers 84 --allow-extra-contacts \
  --reuse-contact-run output/taxonomy_contact_filter_shadow/runs/cf588a9b1a0675b6
```

复用前逐组核对源文件身份、行号、场景配置、接触阈值和模板定义，恢复结果写入新运行目录。
原始库和旧严格接触审计保留；数据集索引在每套新结果完成后更新。
旧严格 run 的抓取输出与对应断点已于 2026-11-12 清理，保留的 9,253 份 `records/*.npz`、
`settings.json` 和历史 `progress.json` 足以继续复用接触审计；当前恢复结果和去重结果不受影响。

默认覆盖 `graspnet_single`、`shapenet_single`、`shapenet_clutter`、
`graspnet_test_clutter`、`graspnet_clutter`，共 74,239,987 条输入记录。
不同场景中的同一源抓取按场景记录分别重放。

```bash
watch -n 5 cat work_runs/taxonomy_contact_filter_shadow/progress.json
```

进度包括已处理组数、行数、保留／改标／歧义／剔除计数，以及各数据集结果路径。
脚本按物体及尺度、或完整场景分组，一个进程复用一个模型处理组内全部类型和目标。
同一命令可恢复已完成组，检查源数据、场景配置、模板与几何模型签名及输出文件状态。
对主进程发送 `SIGTERM` 会记录中断状态并关闭其工作进程。

## 结果组织

```text
work_runs/taxonomy_contact_filter_shadow/
├── progress.json
└── runs/<run_id>/
    ├── settings.json
    ├── progress.sqlite
    ├── progress.json
    ├── transitions.json
    └── <source_dataset_id>/
        ├── filtered_grasps/<assigned_taxonomy>/...
        ├── ambiguous_grasps/<original_taxonomy>/...
        └── records/<object_scale_or_scene>.npz
```

`filtered_grasps` 保存保留与唯一重归类的三阶段姿态。`source_grasp_paths` 和
`source_indices` 指向原 NPY 文件及其行号，原始抓取完整保留。

上面的目录树表示当前完整结果。旧严格 run `cf588a9b1a0675b6` 现在仅是审计来源，
其 `progress.json` 保持历史原值；`complete` 表示接触审计已完成，不表示旧抓取文件仍保留。
`original_taxonomy_ids`、`matching_taxonomy_masks`、`contact_body_masks` 分别记录来源类别、
全部匹配类别和实际接触部件；对应位序在 `taxonomy_names`、`taxonomy_contact_body_names` 中。
原始锚点编号改存为 `source_hand_contact_indices`，避免重归类后误认为它属于新类别模板。
当前结果另存 `strict_matching_taxonomy_masks`、`original_label_restored`、
`extra_contact_body_masks`、`missing_required_group_masks`，区分旧严格匹配与本次恢复。
`matching_taxonomy_masks` 对恢复行增加原类别位；其他候选仍为旧严格规则的候选，
它不表示遍历了“允许额外接触”条件下的全部类别。

`ambiguous_grasps` 的目录名仅表示原类别，候选类别由匹配位掩码给出。
NPZ 审计文件还记录每个源文件的行范围、逐行状态、原类别缺失的必需组、额外部件等。
未通过的姿态不复制到新结果，仍可通过审计及原始库复查。

每套数据全部处理并核对输入文件数与行数后，才自动在 `output/dataset_index.json`
登记 `taxonomy_filtered_<source_dataset_id>`，重新打开控制面板后会显示“接触复验”数据集。
目录放在自定义的 `output/` 之外或使用诊断限额时，不登记为正式数据集。

诊断参数：`--datasets` 选择输入库，`--group` 选择物体尺度或场景，
`--max-groups`／`--max-per-file` 限量测试，`--dry-run` 只统计输入选择。

## 查看未允许接触的例子

通用查看器增加 `--show-contacts`：按当前文件的 taxonomy 重放目标接触，
绿色标记模板允许的实际接触部位和点，红色标记模板外的额外部位和点。
当前允许额外接触，红色本身不再是恢复抓取的剔除依据。
接触球为便于观察进行了放大；判定仍使用过滤时的 2 mm 阈值，正间距不表示穿透。
复查本次结果使用相同的 MuJoCo 3.3.3 环境。

已从原始库提取物体 005、031、048 的三个失败抓取，保存在
[可视化样例](../visualizations/disallowed_contacts_20261112/examples/)。
它们均满足原类的必需接触组，但各多出一个模板外的手部部件，且不符合其他类别的严格规则。
这三个例子均属于当前应恢复的抓取；图中“未允许”指原模板标注范围。
[对照图](../visualizations/disallowed_contacts_20261112/disallowed_contacts.png)
提供完整视图和物体半透明的接触局部；
[来源清单](../visualizations/disallowed_contacts_20261112/examples.json)
记录原始文件、行号、审计位置和实际间距。

```bash
.envs/mujoco333/bin/python scripts/view_clutter_grasps_mujoco.py \
  --root visualizations/disallowed_contacts_20261112/examples \
  --max-grasps 1 --show-contacts --distance 0.42
```

左右方向键切换例子，`H` 显示／隐藏手，`Q` 退出。
添加 `--render-dir <目录>` 可导出各条目的 PNG 及接触摘要 JSON。

## 当前仍剔除的抓取（2026-11-12 复查）

重新扫描当前运行 `094006d47697de2e` 的全部 9,253 份审计，仍剔除 193 条：
GraspNet 训练场景 189 条、GraspNet 单物体 1 条、ShapeNet 单物体 3 条。
另外 2 条 ShapeNet 单物体记录仍待定，不计入剔除。
必需接触组缺失分别为：拇指末节 178 条、小指末节 13 条、无名指组 1 条、手掌／小指掌骨组 1 条。

使用相同的 MuJoCo 3.3.3 逐条重放了全部 193 条，接触掩码与保存审计完全一致。
通过 `mj_geomDistance` 查询缺失组碰撞几何到目标的最近距离，范围为
2.000000039–2.000012133 mm，即只比 2 mm 阈值大约 0.04–12.13 nm，属于数值阈值边界情况。
这些记录对应 7 条单物体来源记录，其中物体 040 的两条记录三阶段姿态完全相同，
去重后共 6 种物体抓取姿态。本次仅统计和可视化，当前保留／改标／待定／剔除结果均未改变。

- [完整统计及来源](../reports/current_rejected_grasps_20261112/summary.json)
- [193 条剔除清单及最近距离](../reports/current_rejected_grasps_20261112/rejected_grasps.csv)
- [全部 193 条重放与距离核对](../reports/current_rejected_grasps_20261112/surface_gap_replay.json)
- [全部 6 种去重姿态对照图](../visualizations/current_rejected_grasps_20261112/current_rejected_grasps.png)

查看器的接触叠加现将缺失的必需接触组标为黄色，并显示最近距离。
黄色组包含多个部件时，原规则要求其中任一部件接触即可；黄色圆圈标的是最近位置。
红色仍表示模板外的额外接触，绿色表示模板允许的实际接触。

```bash
.envs/mujoco333/bin/python scripts/view_clutter_grasps_mujoco.py \
  --root visualizations/current_rejected_grasps_20261112/examples/all \
  --max-grasps 1 --show-contacts --distance 0.55
```

左右方向键切换全部 193 条实例。`examples/representatives` 则只包含 7 条来源记录的代表场景。
