# DexGraspNet 2.0：219 个 ShapeNet 测试场景抓取过滤

这是独立于单物体抓取生成的第二阶段。输入为已完成的 937 个物体单物体抓取库，以及官方公开测试场景的摆放标注；逐场景变换并筛选已有抓取，不重新生成抓取。

## 数据与坐标

- 场景压缩包：`downloads/dexgraspnet2_acronym_test_scenes.tar.gz`，使用资产导入器固定的官方版本及 SHA256。
- 场景分为 `test_acronym_dense`（100 个）、`test_acronym_random`（89 个）、`test_acronym_loose`（30 个），共 219 个场景，包含 937 个唯一物体、1,534 个场景内目标。
- 物体资产：`assets/object/dexgraspnet2_shapenet`，继续使用生成单物体抓取时的米制原始尺度和碰撞网格。
- 单物体抓取库：`output/single_object_grasps_dgn2_shapenet_shadow/succ_collect`。

位姿转换沿用[官方评测代码](https://github.com/PKU-EPIC/DexGraspNet2/blob/26ecd76121e3c8218ad53db9840cf34f6b81b076/src/eval/evaluate_dexterous.py#L80-L97)：标注四元数为 `xyzw`，转成 MuJoCo 的 `wxyz`；平移的高度减去 0.05 米，桌面位于 $z=0$。不对物体再次缩放或重新摆放。

设标注平移为 $t_a$，旋转为 $R_a$，则场景中的物体位姿为：

$$
T_{WO}=\begin{bmatrix}R_a&t_a-(0,0,0.05)^T\\0&1\end{bmatrix}.
$$

若单物体库中的物体位姿为 $T_{SO}$、手部位姿为 $T_{SH}$，转换后的手部位姿为：

$$
T_{WH}=T_{WO}T_{SO}^{-1}T_{SH}.
$$

只变换手腕的平移和旋转，保留手指关节角。预抓取、抓取和夹持目标使用相同变换。

## 执行

先导入场景；脚本会校验压缩包、物体 ID 和场景清单，并逐场景编译完整 MuJoCo 模型：

```bash
PYTHONPATH=. /opt/conda/bin/python scripts/prepare_dexgraspnet2_test_scenes.py
```

再过滤全部 219 个场景，默认 84 个 CPU 工作进程、全部 33 类抓取、不限制候选数量：

```bash
PYTHONPATH=. MUJOCO_GL=egl /opt/conda/bin/python -u scripts/filter_dexgraspnet2_scene_grasps.py
```

VS Code 对应启动项为 **DexGraspNet 2.0：接入 219 个 ShapeNet 测试场景** 和 **5. DexGraspNet 2.0：219 个 ShapeNet 测试场景抓取过滤**。

过滤复用 [GraspNet 场景过滤器](graspnet_scene_filter_zh.md)：每个进程加载一个目标对应的完整静态场景，遍历其全部抓取类型，检查预抓取、8 段接近路径、夹持姿态的桌面及邻近物体碰撞、接近过程中目标物体穿透和最终目标接触。单物体库已做动力学验证；场景阶段默认不做额外抬升仿真（`lift=false`），也不做机械臂轨迹规划。需要额外抬升验证可传 `--lift`，这会创建另一套独立结果。

## 结果与进度

```text
assets/scene/dexgraspnet2_shapenet_test/
  raw/acronym_test_scenes/test_acronym_*/scene_*.npz
  scene_cfg/scene_{dense,random,loose}_*/<物体ID>.npy
  scenes/<场景名>.json
  import_manifest.json
  verification.json

output/dexgraspnet2_scene_filter_shadow/
  progress.json
  runner.log
  runs/<配置指纹>/
    progress.json
    progress.sqlite
    filtered_grasps/<抓取类型>/<场景名>/<物体ID>.npy
```

`progress.json` 的 `finished_targets / targets` 表示完成的场景内目标数；`counts.evaluated_count` 为已完成目标内的候选检查数，`counts.valid_count` 为保留抓取数。单个场景含多个目标，因此目标数量大于 219。进度按完整目标提交，运行中工作进程内的检查尚未计入。

同参数重跑从 SQLite 检查点恢复；源抓取或场景变化会使对应目标重新过滤。输出目录与已有 GraspNet 训练场景过滤结果分离，不修改原单物体抓取库。
