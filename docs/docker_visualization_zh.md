# Dexonomy Docker、Dev Container 与官方可视化

本配置直接使用仓库已有的两种官方可视化操作，不提供额外网页：

- `op=v3d`：每条数据导出手与物体的 OBJ Mesh，可用 Blender、MeshLab 等工具查看。
- `op=vusd`：将多条抓取及多个姿态帧写入 USD，可用 `usdview` 或 NVIDIA Omniverse 查看。

## 1. 初始化官方手模型

Shadow Hand 的 Mesh 来自 MuJoCo Menagerie：

```bash
git submodule update --init --progress third_party/mujoco_menagerie
```

BODex 用于可选的无碰撞机械臂轨迹合成，OBJ/USD 可视化不需要它。

已生成的五套数据统一使用一个 MuJoCo 查看器：

```bash
# 控制面板：选择数据集、grasp type、物体／尺度或场景，以及采样数量。
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py --gui

# 也可以继续直接指定参数：
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py \
  --dataset graspnet_test_clutter --scene scene_0100 --max-grasps 5
# 单物体结果：
MUJOCO_GL=glfw python scripts/view_clutter_grasps_mujoco.py \
  --dataset graspnet_single --object 031 --limit 1 --max-grasps 20
```

`--dataset` 从 `output/dataset_index.json` 读取路径；`--root` 可直接指定 `succ_collect` 或 `filtered_grasps`。快捷键：右方向键/`D` 下一场景或尺度，左方向键/`A` 上一个，`1/2/3` 切换 pregrasp/grasp/squeeze，`H` 显示或隐藏手，`Q`/Esc 退出。`--check` 只加载和编译数据，不打开窗口。不同尺度分别显示，FPS 只影响显示数量。

VS Code 的 **MuJoCo：通用数据集查看器** 启动项直接打开控制面板。
在面板中依次选择数据集、抓取类型和物体或场景，设置“每物体最多”的数量，点击
**打开 / 更新抓取可视化**。更改参数后再次点击该按钮，会关闭上次由面板打开的窗口并
加载新选择；**关闭 MuJoCo 窗口** 保留面板，关闭面板也会关闭它启动的窗口。
面板只列出有抓取结果的条目，因此可选场景数可能小于数据集包含的全部场景数。

物体编号与尺度分开选择，尺度从对应场景配置读取，显示为“原始尺度 · 1.00×”、
等比倍数或 X/Y/Z 各方向倍数。当前两套单物体库每物体仅保存一个原始尺度，因此
尺度栏禁用并明确标注，不显示 `scale_<hash>` 文件名。右侧支持拖动旋转和滚轮缩放的
模型预览，同时显示按实际缩放计算的网格包围盒尺寸（厘米）；单物体使用模型坐标，
场景使用所有物体在场景坐标下的整体范围。预览用于辨认物体，抓取在 MuJoCo 中显示。
姿态与不透明度位于“更多显示设置”，运行日志默认收起。

桌面面板使用 Qt，当前容器已经配置。新环境安装可选依赖：

```bash
python -m pip install -e '.[viewer]'
```

Dockerfile 包含 Qt 所需的 X11 库、Fontconfig 和 Noto 中文字体；重建镜像会一并安装。
命令行查看与 `--check` 不依赖 Qt。

采样数量按每个目标物体分别计算，0 表示全部；不足指定数量时显示全部。
FPS 用抓取时的手掌位置做最远点采样，随机采样不放回；改变随机种子可以换一组。
采样仅影响显示，不写回抓取数据。命令行可用 `--sampling fps` 或
`--sampling random`、`--sample-seed 12`，`--item` 可指定面板列出的完整物体／尺度条目。

如果程序运行却看不到窗口，先看启动输出中的 `Graphics display: DISPLAY=...`。
容器可能继承登录界面的 `:0`，而用户桌面实际上是 `:1`。查看器会在只有一个
当前用户拥有的本地 X11 显示器时自动选择它；SSH 转发地址保持原值。也可以显式
添加 `--display :1` 指定桌面（请使用宿主桌面的实际编号）。X11 socket 和授权文件
仍需挂载，`--display` 本身不会建立远程桌面或绕过授权。

## 2. 根据 README 解包数据

```bash
python scripts/prepare_data.py --group core
```

脚本的数据映射如下：

| 压缩包 | 解包位置 |
| --- | --- |
| `DGN_5k_processed.tar.gz` | `assets/object/DGN_5k/` |
| `DGN_5k_vision.tar.gz` | `assets/object/DGN_5k/` |
| `Dexonomy_GRASP_shadow.tar.gz` | `data/extracted/Dexonomy_GRASP_shadow/` |

也可分别使用 `--group processed`、`--group vision` 或 `--group grasps`。完成标记保存在 `data/.dexonomy_extracted/`，重复运行会跳过已完成的压缩包；`--force` 可强制覆盖。

GraspNet 的全部 88 个物体通过 `python scripts/prepare_graspnet_objects.py --download`
下载到 `downloads/` 并接入 `assets/object/graspnet_1billion/`。
当前单物体流程见 [DGN 与 GraspNet 单物体抓取生成](single_object_grasps_zh.md)。

## 3. 构建 Docker 环境

```bash
docker compose build dexonomy
```

大数据通过卷挂载，不会复制进镜像。CPU 工具使用 `dexonomy` 服务；需要 GPU 的合成任务使用 `dexonomy-gpu`。

## 4. 使用官方 `v3d` 导出 OBJ

输出位于：

```text
visualizations/official_visualization/v3d/succ_grasp/
├── 1_Large_Diameter/
│   └── <object>/<scene>/scaleXXX_hand.obj
└── ...                                   scaleXXX_obj.obj
```

直接使用官方命令：

```bash
docker compose run --rm dexonomy dexrun \
  op=v3d hand=shadow op.data=succ_grasp op.n_max=5 n_worker=4 \
  tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_3d_dir=visualizations/official_visualization/v3d
```

可继续使用官方选项，例如 `op.hand.mode=collision`、`op.hand.skeleton=True`。
用 `tmpl_name=7_Prismatic_3_Finger op.n_max=1 debug_name=63d7e193` 可筛选指定物体。

## 5. 使用官方 `vusd` 导出 USD

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=visualizations/official_visualization/vusd
```

默认写入最终抓取姿态。若需要把预抓取、抓取和挤压阶段连续写入 USD：

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  'op.qpos=[pregrasp,grasp,squeeze]' \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=visualizations/official_visualization/vusd
```

输出为 `visualizations/official_visualization/vusd/usd/1_Large_Diameter.usd`，打开方式：

```bash
usdview visualizations/official_visualization/vusd/usd/1_Large_Diameter.usd
```

若要按数据集文件名顺序导出某个抓取分类的全部样本，请将 `n_max` 设为 `0`，并使用分块输出避免单个 USD 过大：

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=0 op.chunk_size=100 op.start=0 \
  n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=visualizations/official_visualization/vusd
```

输出文件形如 `1_Large_Diameter_000000_000099.usd`，可按文件名依次打开。每个 USD 旁边的 `.manifest.tsv` 记录帧区间与原始数据文件的对应关系。可通过 `op.start=1000` 从排序后的第 1000 个样本继续导出。

## 6. VS Code Dev Container

1. 安装 VS Code 扩展 **Dev Containers**。
2. 打开仓库，执行命令面板中的 **Dev Containers: Reopen in Container**。
3. 首次进入会构建同一个 `dexonomy-runtime:local` 镜像并挂载整个工作区。

容器内默认解释器为 `/opt/conda/bin/python`，并预装 Python、Pylance、Docker、YAML 与 Ruff 扩展。VS Code 会把容器内 `vscode` 用户的 UID/GID 自动映射到宿主用户，避免生成 root 所有的文件。Dev Container 默认请求全部 NVIDIA GPU，因此宿主机需要 NVIDIA Container Toolkit。若只做 CPU 可视化，请删除 `.devcontainer/docker-compose.devcontainer.yml` 中的 `deploy.resources` 段。

## 7. GPU 合成

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm dexonomy-gpu dexrun op=tmpl
docker compose run --rm dexonomy-gpu dexsyn 'tmpl_name=[1_Large_Diameter]'
```

`assets/` 与 `output/` 可写，`data/` 保持只读。默认镜像不包含可选 BODex；若需 UR10e + Shadow Hand 的无碰撞轨迹合成，请继续按主 README 安装 BODex。

## 8. MuJoCo 4.0 升级说明

项目依赖固定为 `mujoco==4.0.0`。MuJoCo 4.0 的 Python 包拆分为
`mujoco-core`、`mujoco-studio` 和统一安装入口 `mujoco`；从旧版本升级时，
先卸载旧入口，避免跨包移动的文件被安装器删除。参见
[官方安装说明](https://mujoco.readthedocs.io/en/latest/python.html#installation)。

在有写权限的 Python 环境中运行：

```bash
python -m pip uninstall -y mujoco
python -m pip install -e .
python -c 'import mujoco; print(mujoco.__version__, mujoco.mj_versionString())'
```

容器镜像通过 `pyproject.toml` 获取同一版本，重建镜像即可应用。现有
`mujoco.viewer`、GLFW 窗口和 EGL 离屏渲染入口继续使用；更新引擎不会自动
把现有查看器切换为 Studio，也不会切换物理计算后端。

代码已适配模型元素删除接口、关节多项式阻尼字段，以及扩展后的
`mjNEQDATA`；焊接约束的新字段保留引擎默认值，已有配置中的线性阻尼不变。
`tests/test_mujoco_compatibility.py` 检查模型编译与步进、阻尼、以及保存的
三阶段姿态在查看器中的重放。

升级保留已有抓取库与验证标签。版本间碰撞和求解器行为可能改变，历史标签
应保留其原验证版本，不能因引擎升级就当作在 4.0 下重新验证成功。2026-11-11
的代码测试、场景加载、FPS 渲染和抽样受力复验记录保存在
`reports/mujoco4_upgrade_20261111/`。这次抽查的 4 条历史抓取在 3.3.3 与 4.0
下均未通过重新受力验证；该小样本不能估计整库成功率，也未覆盖或修改原标签。

当前工作区保留独立的旧版本复现入口，可用相同命令替换 Python 路径：

```bash
.envs/mujoco333/bin/python -c 'import mujoco; print(mujoco.__version__)'
```

这个目录是本地回退环境，不随仓库分发；新环境需自行建立对应虚拟环境。
