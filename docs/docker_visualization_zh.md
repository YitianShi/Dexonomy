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

若希望不经过 OBJ/USD、直接用 MuJoCo Viewer 按顺序浏览发布数据集：

```bash
MUJOCO_GL=glfw python scripts/view_dataset_mujoco.py \
  --taxonomy 1_Large_Diameter --start 0 --limit 0 --pose-seconds 0.5
```

Viewer 按排序后的数据文件逐个加载，并依次显示文件中的全部 `grasp_qpos`。快捷键：右方向键/`D` 下一个样本，左方向键/`A` 上一个样本，`N`/`P` 切换姿态，空格暂停或继续，`Q`/Esc 退出。在已配置 X11/XWayland 的本机或 Dev Container 内运行此命令。

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
output/official_visualization/v3d/succ_grasp/
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
  vis_3d_dir=output/official_visualization/v3d
```

可继续使用官方选项，例如 `op.hand.mode=collision`、`op.hand.skeleton=True`。
用 `tmpl_name=7_Prismatic_3_Finger op.n_max=1 debug_name=63d7e193` 可筛选指定物体。

## 5. 使用官方 `vusd` 导出 USD

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=output/official_visualization/vusd
```

默认写入最终抓取姿态。若需要把预抓取、抓取和挤压阶段连续写入 USD：

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  'op.qpos=[pregrasp,grasp,squeeze]' \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=output/official_visualization/vusd
```

输出为 `output/official_visualization/vusd/usd/1_Large_Diameter.usd`，打开方式：

```bash
usdview output/official_visualization/vusd/usd/1_Large_Diameter.usd
```

若要按数据集文件名顺序导出某个抓取分类的全部样本，请将 `n_max` 设为 `0`，并使用分块输出避免单个 USD 过大：

```bash
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=0 op.chunk_size=100 op.start=0 \
  n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=output/official_visualization/vusd
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
