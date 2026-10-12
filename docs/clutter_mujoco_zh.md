# 使用 MuJoCo 生成 cluttered scene 并过滤抓取

这条流水线只使用 MuJoCo，不依赖 PyBullet、Isaac Gym 或 USD。实现采用了两类已有方法中的核心设计：

- `vmf-contact2`：随机选择物体、在桌面上方投放、等待连续稳定、拒绝越界或超时场景，并在抓取与预抓取位置检查完整场景碰撞。
- [DexGraspNet 2.0](https://github.com/PKU-EPIC/DexGraspNet2)：先复用单物体抓取标签，再映射到 clutter 场景，最后通过物理仿真筛选。

## 1. 一次完成场景生成与抓取过滤

先构建包含当前代码的镜像：

```bash
docker compose build dexonomy
```

默认生成 5000 个场景，每个场景随机包含 4–8 个物体，对每个物体各写一份 target scene config，并立即过滤源数据中的全部抓取类型（`op.filter.taxonomies=null`）。物体尺度采样也使用这些类型的抓取数据。每个编号执行“生成或读取已有场景 → 过滤全部 target → 检查抓取总数”的循环。整个场景在启用的 taxonomy 中合计至少保留一条抓取才完成；不要求每个物体都有抓取。总数为 0 时，删除该编号的场景及所有 taxonomy 下的抓取输出，重新采样物体、尺度和投放姿态，再生成并过滤，直到成功。桌面穿透也会重新采样并保留原编号；正常完成后编号连续：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=all
```

根参数 `n_worker` 控制 CPU 进程数，默认 32。每个进程独立完成一个编号的生成、过滤及必要的重建，再领取下一个编号；任务完成顺序可以不同，manifest 仍按场景编号排序。已有场景直接过滤，不会因已保存 scene config 而跳过抓取检查。`skip_existing` 仅控制 `op.stage=scene` 的纯生成任务。

工作进程优先使用 `forkserver`，在独立服务进程中预加载 CPU 依赖；不支持时回退到 `spawn`。物体目录只序列化一次，子进程从本地临时文件并发读取；每个进程的原生计算线程限制为 1，避免进程数与 BLAS/OpenMP 线程数相乘。场景生成入口不加载 Torch，抓取过滤需要的手部仿真依赖在第一次过滤时才加载。日志会分别显示物体目录扫描耗时、每个进程的 `worker ready`、过滤尝试次数及最终抓取数。

当前 VS Code 入口只生成独立物体抓取，见 [单物体流程](single_object_grasps_zh.md)。场景流程保留为后续命令行功能。

后续场景流程可通过 `op.scene.spawn_xy_scale` 和 `op.scene.spawn_gap` 控制投放范围与间隙；已有场景不受新投放参数影响。

输出结构：

```text
output/clutter_mujoco_shadow/clutter/
├── scene_cfg/scene_000000/<target_object>.npy
├── filtered_grasps/<taxonomy>/scene_000000/<target_object>.npy
└── manifest.json
```

`manifest.json` 记录每个目标的输入数、尺度不匹配数、场景碰撞数、无目标接触数、抬升失败数和最终保留数。完整循环的 `scene_records` 还包含 `valid_grasp_count`、`filter_attempts` 和 `reused_existing`。重新生成过的场景包含 `regeneration_count` 和 `regeneration_failures`，其中 `zero_valid_grasps` 表示零抓取导致的重建次数；只保留最终成功场景及其过滤记录。

## 2. 分阶段执行

只生成稳定场景：

```bash
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=scene \
  op.scene.num_scenes=100 n_worker=8
```

过滤已经生成的场景，若整个场景为零抓取则按原编号重建并继续过滤：

```bash
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=filter n_worker=8
```

全部 taxonomy 默认已启用。若只需过滤指定类型，可覆盖为列表：

```bash
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=filter \
  'op.filter.taxonomies=[1_Large_Diameter,2_Small_Diameter]'
```

额外启用严格的 squeeze-and-lift 物理验证：

```bash
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=filter \
  op.filter.lift.enabled=true
```

## 3. 过滤规则

每条单物体抓取先通过目标物体的刚体变换映射到稳定场景，然后执行：

1. 对 `pregrasp → grasp` 做位置和四元数插值；默认不检查精确的 `pregrasp` 起点，但仍检查其后的全部插值点和 `grasp`；
2. 检查全部接近姿态以及 `squeeze` 姿态；
3. 允许接触当前 target，拒绝手与桌面或其他物体穿透；
4. 在 `squeeze` 姿态要求存在 target 接触；
5. 可选启用 MuJoCo 重力、摩擦系数 0.2 和 0.1 kg 物体质量，执行 squeeze-and-lift，目标至少升高 5 cm 才保留。

默认关闭第 5 步，因为 `succ_collect` 本身已经是经过单物体物理验证的发布数据；场景级过滤默认复现 `vmf-contact2` 的接近/抓取碰撞检查。需要更严格的场景内动力学复验时设置 `op.filter.lift.enabled=true`。

发布版 `succ_collect` 中的 `scene_scale` 是每条抓取相对于 source scene 物体 scale 的倍率。过滤器会合并同一物体的 source scale variants，并用 `source object scale × scene_scale` 得到抓取对应的实际物理尺度；默认要求它与 clutter scene 的固定物体尺度相差不超过 2%。默认跳过精确的 pregrasp 起点，从第一个插值点到 grasp 不允许手对目标物产生超过 1 mm 的穿透；squeeze 是闭合控制目标，因此不套用这项静态穿透限制。

场景生成默认启用 `op.scene.sample_dataset_scales=true`。每次选择物体时，会从当前启用 taxonomy 的 `succ_collect` 中随机抽取一条真实抓取，并把物体设置为该条抓取的有效尺度 `source object scale × scene_scale`。所选 taxonomy、源文件、源索引、倍率和最终尺度都会写入 scene config 的 `clutter.selected_grasp_scales`；后续过滤再测试该尺度附近的全部候选。设置为 `false` 可恢复只从 source scene config 随机选择名义尺度的旧行为。

### 实时查看场景生成过程

在已配置 X11 的本机或 Dev Container 中直接运行，可实时查看物体出生、下落、碰撞和稳定判定：

```bash
MUJOCO_GL=glfw python -m dexonomy.main op=clutter exp_name=clutter_debug \
  op.stage=scene op.seed=null op.scene.num_scenes=10 \
  op.scene.skip_existing=false op.scene.debug_view=true
```

此命令启用 GLFW、`debug_view=true` 和实时播放，连续生成 10 个调试场景
（`scene_000000` 到 `scene_000009`），输出到 `output/clutter_debug_shadow/`。
当前 VS Code 启动配置只提供单物体流程，场景调试使用上述命令。

调试命令设置 `op.seed=null`，每次启动使用新的随机种子，重新采样物体、尺度和投放姿态。实际种子会输出到终端并写入 `clutter/manifest.json` 的 `seed` 字段；重新生成同编号场景时，会清理该场景目录中不再使用的旧 target 配置。普通生成任务仍默认使用固定种子 36。

指定场景数量、起始编号或调整物体数量：

```bash
MUJOCO_GL=glfw python -m dexonomy.main op=clutter exp_name=clutter_debug \
  op.stage=scene op.seed=null op.scene.skip_existing=false op.scene.debug_view=true \
  op.scene.num_scenes=20 \
  op.scene.start_index=12 \
  op.scene.objects_min=5 \
  op.scene.objects_max=5
```

要复现某次运行，把记录的种子作为 `op.seed=<seed>` 传入；VS Code 调试时可将 `launch.json` 中的 `op.seed=null` 替换为该数值。

随机采样现在为每个场景编号派生独立随机种子，因此同一 `op.seed` 和编号在不同 `n_worker` 或断点续跑时会得到相同的采样结果；新版本随机序列与旧串行版本不同。开启 `op.scene.debug_view=true` 时自动使用单进程，避免同时打开多个 Viewer。

Viewer 中按 `Space` 暂停/继续，暂停时按 `N` 单步。`debug_realtime=true` 默认按仿真时间实时播放；可通过 `op.scene.debug_spawn_pause` 和 `op.scene.debug_result_pause` 调整出生姿态及判定结果的停留时间。

## 4. 关键配置

- `op.scene.objects_min/max`：每个场景的物体数。
- `op.scene.table_size`：桌面 XY 尺寸，默认 0.4 m × 0.4 m，正方形桌面。
- `op.scene.table_friction`：桌面滑动摩擦系数，默认 `1000.0`，用于近似无限大摩擦；物体之间的摩擦由 `op.scene.friction` 控制，默认 `0.8`。桌面摩擦也写入 scene config，后续 MuJoCo 加载时会应用。
- `op.scene.impratio` / `op.scene.noslip_iterations`：默认 `10.0` / `3`，配合椭圆摩擦锥抑制桌面接触的数值滑移。参见 [MuJoCo 防滑设置](https://mujoco.readthedocs.io/en/3.3.3/modeling.html#preventing-slip)。
- `op.scene.spawn_xy_scale`：初始投放区域相对桌面 XY 的比例，默认 `1.0`，覆盖整个 0.4 m × 0.4 m 桌面；采样时仍保留物体尺寸所需的边缘距离。设置 `0.7` 可恢复中央聚集区域。
- `op.scene.spawn_gap`：初始物体包围球之间的三维间隙，默认 `0.02 m`（2 cm）；该参数不保证落稳后的水平间距。
- `op.scene.stable_time`：速度持续低于阈值的时间，默认 0.5 秒。
- `op.scene.linear_velocity`：结束落体仿真的线速度阈值，默认 `0.0001 m/s`（0.1 mm/s）；全部物体的线速度严格低于该阈值，并连续保持 `stable_time` 后才判定稳定。
- `op.scene.settle_timeout`：单次落体最长仿真时间，默认 15 秒。
- `op.scene.table_penetration_tolerance`：桌面穿透容差，默认 `0.001 m`（1 mm）。初始姿态检测到桌面穿透、仿真中物体完全落到桌面以下，或落稳后与桌面的穿透深度超过容差时，会丢弃当前候选并保留原场景编号，重新随机选择物体、尺度和投放姿态，直到该编号成功。穿透后不再重投同一组物体，也不会因连续穿透达到 `max_scene_attempts` 而放弃编号。落体碰撞时短暂的软接触压缩不会直接触发重采样。
- `op.scene.max_object_speed`：与 vmf-contact2 一致，每个物理步后把物体线速度限制到 0.5 m/s；只缩放线速度向量，不修改角速度，设为小于等于 0 可关闭。
- `op.scene.freejoint_damping`：落体阶段的临时自由关节阻尼，用于抑制细长物体持续旋转。
- `op.scene.containment_walls`：落体阶段启用临时不可见围挡；围挡不会写入最终场景。
- `op.scene.spawn_position_attempts`：使用包围球采样无初始重叠位置的最大次数。
- `op.scene.targets_per_scene=0`：为场景内每个物体生成 target 配置。
- `op.filter.approach_steps`：接近路径的插值检查数。
- `op.filter.check_pregrasp`：是否检查精确的 pregrasp 起点，默认 `false`；后续接近路径仍然检查。
- 根参数 `n_worker`：CPU 工作进程数，默认 32。例如 `n_worker=8` 同时处理最多 8 个场景，各自生成后立即过滤；同一场景的 target 在该进程内依次检查。实际进程数不会超过场景任务数；`n_worker=1` 或开启生成 Viewer 时使用单进程。
- `op.filter.max_per_target`：每个 target/taxonomy 在碰撞过滤前最多验证多少条，0 表示全部；它不控制最终保存数量。
- `op.filter.lift.enabled`：是否执行最终 MuJoCo 抬升验证，默认关闭。

## 5. 用 MuJoCo Viewer 查看结果

在一个窗口中加载完整 clutter scene，并同时显示该场景下所有 target 的已过滤抓取：

```bash
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py \
  --root output/clutter_mujoco_shadow/clutter/filtered_grasps
```

`--taxonomy` 默认为 `all`，因此会合并该场景所有 taxonomy、所有 target 的抓取；
显示数量由 `scripts/clutter_grasp_viewer.yaml` 中的 `max_grasps_per_target` 控制，默认值为 5。Viewer 在 grasp 阶段的掌根三维位置上执行最远点采样（FPS），并按 target 物体独立采样：每个物体在场景中的当前 scale 最多显示 5 条，而不是整个场景合计 5 条；生成和过滤后的完整抓取数据不会被修改。因此，一个包含 4 个有有效抓取物体的场景最多显示 20 条。将 YAML 中的值改为 `0` 可显示全部抓取；`sample_seed` 控制每组 FPS 的第一个采样点。命令行 `--max-grasps` 和 `--sample-seed` 可临时覆盖 YAML。也可以只查看某个 taxonomy 并调整透明度：

```bash
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py \
  --root output/clutter_mujoco_shadow/clutter/filtered_grasps \
  --taxonomy 1_Large_Diameter --scene scene_000001 --hand-alpha 0.25
```

快捷键：`1` 显示 pregrasp，`2` 显示 grasp，`3` 显示 squeeze，`H` 隐藏/显示全部手，左右方向键（或 `A/D`）切换场景，`Q/Esc` 退出。不同 target 的抓取使用不同颜色。

仅检查数据和 MuJoCo 模型能否正确编译，不打开窗口：

```bash
docker compose run --rm -T \
  -v "$PWD/scripts/view_clutter_grasps_mujoco.py:/tmp/view_clutter_grasps_mujoco.py:ro" \
  -v "$PWD/scripts/clutter_grasp_viewer.yaml:/tmp/clutter_grasp_viewer.yaml:ro" \
  dexonomy python /tmp/view_clutter_grasps_mujoco.py \
  --root output/clutter_mujoco_shadow/clutter/filtered_grasps \
  --taxonomy 1_Large_Diameter --check
```

也可以在 VS Code 中直接调试：

1. 执行 **Dev Containers: Rebuild and Reopen in Container**；首次启动会从宿主机复制当前 X11 cookie。
2. 打开“运行和调试”（`Ctrl+Shift+D`）。
3. 选择 **MuJoCo: View clutter grasps** 后按 `F5`，输入 filtered grasp 根目录和 scene 名称。
4. 在 `scripts/view_clutter_grasps_mujoco.py` 中设置的断点会直接生效。若只想检查数据，选择 **MuJoCo: Check clutter grasps (no window)**。

两个调试入口都会读取 `scripts/clutter_grasp_viewer.yaml`，所以显示数量仍由 `max_grasps_per_target` 控制，不影响生成的数据集。宿主机必须设置 `DISPLAY`、`XAUTHORITY` 并安装 `xauth`。

如果只想逐个查看单个 target 文件中的抓取，继续使用原来的顺序查看器：

```bash
MUJOCO_GL=glfw python scripts/view_dataset_mujoco.py \
  --root output/clutter_mujoco_shadow/clutter/filtered_grasps \
  --taxonomy 1_Large_Diameter --limit 0 --qpos grasp
```

场景配置本身也兼容 Dexonomy 的 `load_scene_cfg`、`MuJoCo_OptEnv` 和后续 `op=grasp` / `op=eval` 流程。`gen_grasp` 现在会分别处理 target 接触、distractor/桌面接触和手部自碰撞，避免把旁边物体误当作目标接触。

## 6. 两个独立步骤：单物体抓取库与杂乱场景过滤

当前 VS Code 只提供 DGN 与 GraspNet 的单物体启动项，详见 [单物体流程](single_object_grasps_zh.md)。以下保留后续场景过滤的命令行说明。

### 单物体抓取生成与验证

启动时可输入物体编号，或输入 `*` 处理全部选中物体。
DGN 单物体启动项固定保存到 `output/single_object_grasps_dgn_shadow`，启动时不询问场景或保存位置。
入口为 `scripts/generate_single_object_grasps.py`。每个物体模型和实际尺度只生成一次；
物体位于原点，环境中没有桌面、其他物体或杂乱场景姿态。
默认 1 轮、128 个表面采样点、8 个旋转角度条目，每个物体/尺度/抓取类型共
1,024 条候选。全部候选进入原生模板匹配、MuJoCo 抓取优化和物理验证，
没有初始化数量截断或 FPS。GPU 2 负责匹配，84 个 CPU 工作进程负责优化和验证，
`skip_traj=true`。

DGN 直接读取已有资产配置中的全部物体和六个原有尺度；GraspNet 使用 `000–087` 的原始实际尺寸。单物体生成不依赖杂乱场景清单。

```bash
python scripts/generate_single_object_grasps.py \
  --object-root assets/object/DGN_5k/scene_cfg \
  --output output/single_object_grasps_dgn_shadow \
  --points 128 --inplane 8 --epochs 1 --gpu 2 --workers 84 --no-keep-failed
```

可以加 `--object 物体编号` 或 `--taxonomies 24_Tip_Pinch` 缩小范围，
先加 `--dry-run` 查看去重后的数量。再次运行同一库会跳过已完成组；
增加新的物体或尺度时，已缓存的物体/尺度结果继续复用。
库中其他物体或尺度的缓存保留，但不会自动追加到本次任务。
`progress.json` 的完成组数、尝试数和成功数只统计本次选择；
`library_completed_groups` 和 `library_valid_grasps` 则统计整个库的历史完成结果。
改变采样点数、旋转数或轮数时，使用 `runs/points_<P>_inplane_<R>_epochs_<E>/`
保存新参数的检查点，不覆盖旧采样和验证记录。各次运行的有效抓取仍汇入根目录
同一个 `succ_collect`，历史抓取及其来源保留。根目录 `progress.json` 显示当前任务，
其中 `run_directory` 和 `run_config` 指向实际检查点目录及参数文件。
第二步会检查目标的物理尺度；尺度不匹配的抓取不会使用。
命令行仍支持可选的 `--scale-inventory` 导入额外尺度；它与仅筛选编号的
`--object-inventory` 不同；当前两个单物体启动项都不使用场景清单。

输出如下：

```text
output/single_object_grasps_dgn_shadow/
├── scene_cfg/<object>/floating/scale_<hash>.npy
├── succ_collect/<taxonomy>/<object>/floating/scale_<hash>.npy
├── sampling/<object>/floating/scale_<hash>/epoch_000.npz
├── attempts/<taxonomy>/<object>/floating/scale_<hash>/epoch_000.npz
├── validated/<taxonomy>/<object>/floating/scale_<hash>/epoch_000.npy
├── progress.json
└── runtime.json
```

默认只保留成功候选的完整数组及原始候选编号；全部候选仍会尝试，
失败计数和完成检查点保留。按 Ctrl+C 或 SIGTERM 会停止提交新任务，
等待已提交组保存；再次使用同一库即可续跑。
进度每组完成后更新，可用 `watch -n 2 cat output/single_object_grasps_dgn_shadow/progress.json` 查看。

### 单尺度抓取多样性分析

原始抓取保持完整。以下命令只读取当前任务中已经完成全部抓取类型和轮数的
物体/尺度组合，使用两个 CPU 进程生成分析报告，不占用 GPU：

```bash
python scripts/analyze_grasp_diversity.py --workers 2
```

可加 `--object 物体编号 --scale 0.05` 检查指定尺度。默认分析 `grasp_qpos`，
也支持 `--phase pregrasp` 或 `--phase squeeze`。每个物体和尺度独立分析，
不会把不同尺度混为一个抓取池；正在生成但尚未完成的尺度留给下一次分析。

默认的近重复条件需要同时满足：手根位置距离不超过物体实际 OBB 对角线的
2%、SO(3) 旋转角不超过 10°、22 个手指关节角差的 RMS 不超过 5°，
任一关节角差不超过 15°。四元数 `q` 和 `-q` 按同一旋转处理。
报告还计算严格和宽松两组阈值，检查结论对阈值是否敏感。
局部近重复数量精确计算；大抓取池的全局距离分位数使用 20,000 对固定种子的随机抓取对。

`output/grasp_diversity_dgn_shadow/` 保存总体 `summary.json`、每尺度指标
`object_scale_metrics.csv`、复查清单 `needs_review.csv`、代表抓取索引
`representatives.npz`、详细示例 `examples.json`、图表 `diversity_report.png` 和中文
`report.md`。索引只建立一个独立视图，不删除任何原始抓取。
代表抓取两两不满足默认近重复条件；每个未入选的有限抓取都与一个入选抓取直接接近，
不会用传递连通关系把一整条逐渐变化的抓取链压成一个姿态。

默认筛查标记为：代表抓取少于 64 个、可合并冗余超过 30%、手根空间方向覆盖
少于 48 个近似等面积方向区间中的 8 个，或出现无效手姿态。
这些是工程筛查阈值。物体几何与可抓面可能解释标记；通过筛查也不等于覆盖全部物理可行抓取。
方向覆盖统计手根相对物体中心的位置，不代表物体接触面积覆盖。

可用以下方式读取报告中的代表抓取；如果源库姿态已变化，读取器会要求重新分析：

```python
from scripts.analyze_grasp_diversity import load_representative_grasps
view = load_representative_grasps(
    "output/grasp_diversity_dgn_shadow",
    "<object>/floating/scale_<hash>",
)
qpos = view["qpos"]
```

### 后续可选：杂乱场景抓取过滤

命令行 `op=clutter op.stage=filter` 读取单物体库的 `succ_collect`；当前任务不执行场景生成或过滤。
对每个场景目标，按物体和尺度选取库中的抓取，变换手根部的世界坐标，
保留手指关节角，再检查接近过程、桌面/其他物体碰撞、目标穿透与接触。
第二步仅使用 CPU，不调用模板匹配器或抓取优化器。
零有效抓取的目标保留零结果；输入场景和单物体抓取库都保持只读。

```bash
python -m dexonomy.main op=clutter op.stage=filter n_worker=84 skip_traj=true \
  save_dir=output/clutter_dgn_library_filtered_shadow \
  'op.filter.scene_cfg_glob=output/clutter_dgn_shadow/clutter/scene_cfg/scene_*/*.npy' \
  op.filter.single_object_library=true \
  op.filter.source_grasp_dir=output/single_object_grasps_dgn_shadow/succ_collect \
  op.filter.max_per_target=0 op.filter.pool_scale_variants=true
```

结果保存到 `output/clutter_dgn_library_filtered_shadow/clutter/filtered_grasps/`。
可用 `scripts/view_clutter_grasps_mujoco.py` 指定结果路径查看。
同一物体和尺度换到另一个场景时，只需重新运行第二步；第一步的库直接复用。

旧 `generate_native_clutter_grasps.py` 入口仍保留，其历史输出属于场景条件生成，
没有作为第一步的独立抓取库使用，也不会自动恢复旧任务。
旧 Objaverse 场景、补充抓取与相关可视化已清理；官方抓取库保留。

### GraspNet 的单物体资产

执行 `python scripts/prepare_graspnet_objects.py --download` 导入官方 88 个物体。
它复用官方 CoACD 网格，保留原始米制尺寸并生成 Dexonomy 配置。
GraspNet 抓取固定保存到 `output/single_object_grasps_graspnet_shadow`。
数据准备、生成参数和进度字段见 [单物体流程](single_object_grasps_zh.md)。
