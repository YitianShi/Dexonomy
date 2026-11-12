# Dexonomy

[Project Page](https://pku-epic.github.io/Dexonomy) | [Paper](https://arxiv.org/abs/2504.18829) | [Dataset](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main) | [Learning](https://github.com/JYChen18/DexLearn)

## Local Generated Data

The five final grasp datasets are physical directories under [output/datasets](output/datasets/README.md). Obsolete producer payloads and checkpoints were removed; new generation/filtering work files default to `work_runs/`. Auxiliary results are stored separately in [analysis](analysis/README.md), [visualizations](visualizations/README.md), and [reports](reports/README.md). See the [data catalog](docs/generated_data_zh.md) for dataset counts and entry points.

Local scripts cover asset preparation, grasp generation, scene filtering, and one [MuJoCo dataset viewer](scripts/view_clutter_grasps_mujoco.py). Standalone analysis, plotting, and experiment scripts have been removed; their saved results remain available. The original Dexonomy operations are preserved.

Open the viewer's desktop control panel with `python scripts/view_clutter_grasps_mujoco.py --gui`, or select **MuJoCo：通用数据集查看器** in VS Code. Choose a dataset, grasp type, object or scene, and the per-object sample count using FPS or random sampling, then click **打开 / 更新抓取可视化**. Object scales use actual numeric values and a mesh preview shows dimensions in centimetres. Install the optional Qt interface with `python -m pip install -e '.[viewer]'`; the Docker image includes it.

## Overview

<div style="text-align: center;">
    <img src="img/teaser.png" width=100% >
</div>

Our algorithm synthesizes **contact-rich, penetration-free, and physically plausible** dexterous grasps for:
- Any grasp type
- Any object
- Any articulated hand

All starting from just **one** human-annotated template *per hand and grasp type*.

## Supported Features
- Hands: Shadow, Allegro, Leap, MANO, Unitree_G1
- Object Scenes: Single (floating & tabletop), Clustered (light clutter)
- Object Assets: Rigid objects (ShapeNet & Objaverse), Articulated objects (PartNet)
- Physics Simulator: MuJoCo
- Grasping Trajectory Synthesis: cuRobo integration

TODO: add scene_cfg examples for articulated objects, clutter scenes, and other tasks.


## Installation
Our code is tested on **Ubuntu 20.04** with **NVIDIA RTX 3090** GPUs. 

```bash
git submodule update --init --recursive --progress

conda create -n dexonomy python=3.10 
conda activate dexonomy
pip install -e .
```

BODex provides optional collision-free arm trajectory synthesis for UR10e + Shadow Hand.

```bash
cd third_party/BODex
git lfs pull
# install BODex
pip install -e . --no-build-isolation
# link to dexonomy's object folder
ln -s ../../../../../../assets/object src/curobo/content/assets/object  
cd ...
```

## Quick Start
### 1. Prepare Object Assets 

Download our pre-processed object assets `DGN_5k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy), and organize the unzipped folders as below. 
```
assets/object/DGN_5k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```
Alternatively, you can pre-process your own object assets using [MeshProcess](https://github.com/JYChen18/MeshProcess).

### 2. Synthesize Grasps
Run the following commands to synthesize grasps for a specific grasp type:
```bash
dexrun op=tmpl  # Convert human annotations to valid templates
dexsyn 'tmpl_name=[1_Large_Diameter]' # Run the script for complete synthesis pipeline
```
where `dexrun` is the alias for `python -m dexonomy.main` and `dexsyn` is the alias for `python -m dexonomy.script`.

## Tutorial
For a detailed walkthrough of **template annotation** and **code usage**, please refer to [getting_started](https://github.com/JYChen18/Dexonomy/tree/main/getting_started).

## Docker and Official Visualization

The runtime pins MuJoCo **4.0.0**. When upgrading an existing environment from
3.3.3, uninstall `mujoco` first, then run `python -m pip install -e .`; 4.0 splits
the distribution into core and Studio packages that share the `mujoco` namespace.
See the [upgrade notes](docs/docker_visualization_zh.md#8-mujoco-40-升级说明).

The Docker runtime supports the repository's official `v3d` (OBJ) and `vusd` (USD) visualization operations:

```bash
# Unpack the five core archives into the README-compatible paths.
python scripts/prepare_data.py --group core
docker compose build dexonomy

# Export official OBJ or USD visualization files for one taxonomy.
docker compose run --rm dexonomy dexrun op=v3d hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_3d_dir=visualizations/official_visualization/v3d
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=visualizations/official_visualization/vusd
```

The outputs are written to `visualizations/official_visualization/`. See [docs/docker_visualization_zh.md](docs/docker_visualization_zh.md) for data mapping, direct `dexrun` commands, `usdview` usage, and GPU synthesis commands.

## MuJoCo Clutter Dataset

Generate stable tabletop clutter scenes, transfer released single-object grasps,
filter scene collisions, and optionally validate squeeze-and-lift entirely in MuJoCo:

```bash
docker compose run --rm dexonomy dexrun \
  op=clutter exp_name=clutter_mujoco op.stage=all
```

Each worker generates or reuses one scene and immediately filters its grasps.
Scenes with zero retained grasps are deleted and resampled under the same ID
until at least one grasp remains. Existing scenes go directly to filtering.

See [docs/clutter_mujoco_zh.md](docs/clutter_mujoco_zh.md) for staged commands,
dataset layout, filtering rules, scale handling, and MuJoCo visualization.

The current VS Code workflow focuses on isolated object grasp generation for
**DexGraspNet** and **GraspNet-1Billion**. Their separate entry points are
`scripts/generate_dexgraspnet_single_object_grasps.py` and
`scripts/generate_graspnet_single_object_grasps.py`. Both reuse
`scripts/generate_single_object_grasps.py`, enumerating every hand contact in
33 Shadow hand templates against 128 surface points and 8 rotation entries.
The templates contain 275 contacts in total, giving 24,780,800 candidates for
all 88 GraspNet objects. GPU 2 performs template matching; 84 CPU
workers run native MuJoCo optimization and physics validation, with arm planning
disabled. Each entry fixes its dataset and output directory and defaults to one
object; `--object '*'` processes all objects independently. No clutter scene inventory is required.

Prepare all 88 GraspNet objects from the official DexGraspNet 2.0 model archive:

```bash
python scripts/prepare_graspnet_objects.py --download
```

This verifies the official archive checksum, reuses its CoACD meshes and URDF
transforms, and writes Dexonomy assets under `assets/object/graspnet_1billion`.
GraspNet meshes retain their original coordinates in meters with scale `[1,1,1]`.
DGN uses its existing six asset scales. Generated libraries are saved separately
to `output/single_object_grasps_dgn_shadow` and
`output/single_object_grasps_graspnet_shadow`.
Exhaustive contact checkpoints use a separate `runs/all_contacts_*` directory;
previous successful grasps remain in the published library.
See [the single object workflow](docs/single_object_grasps_zh.md) for launches and progress.

For completed dataset counts, storage paths, provenance and cleanup records, see
the [generated data catalog](docs/generated_data_zh.md).

The separate **3. DexGraspNet 2.0：剩余 ShapeNet 单物体抓取生成与验证**
launch covers the 937 unique ShapeNet objects in the released official test
annotations, excluding the 88 GraspNet objects. Prepare them with
`python scripts/prepare_dexgraspnet2_objects.py --download --workers 24`, then run
`python scripts/generate_dexgraspnet2_single_object_grasps.py`.
This entry defaults to all selected objects, GPU 0 and 84 CPU workers. It keeps
the physical coordinates from the official `scaled.obj`, builds V-HACD convex
collision meshes and validates the MuJoCo assets. Inputs are fixed under
`assets/object/dexgraspnet2_shapenet`; generated grasps are saved separately under
`output/single_object_grasps_dgn2_shapenet_shadow`. The public test inventory has
937 ShapeNet IDs rather than the 1,231 reported in the paper; the importer pins
the official archive revision and records the exact selection and checksums.

To download every model in the bundled model pool, use **DexGraspNet 2.0：下载附带模型库（7,985 个）**
or `python scripts/prepare_dexgraspnet2_objects.py --full-release --download --downloads downloads/dexgraspnet2_full_20261014`.
The full raw model files are kept under `assets/object/dexgraspnet2_full`, with
7,897 complete ACRONYM geometries and 88 GraspNet models indexed separately from
incomplete model directories. This 7,985-model catalog is larger than the paper's
1,319-object benchmark. `benchmark_inventory.json` separately records the 1,025 IDs
confirmed from the released annotations and all 88 GraspNet models; 294 benchmark
IDs remain unresolved. The extra model pool does not complete this mapping.
Extra raw models require MuJoCo asset conversion before generation.

Prepare the official 7,600 training scene layouts separately with
`python scripts/prepare_dexgraspnet2_train_scenes.py --download --workers 3`, or
the **DexGraspNet 2.0：下载并解包 7600 个训练场景（仅场景）** launch.
The importer downloads the nine relevant official archives (124.6 GB in total),
checks their SHA256, and retains only view-0 pose annotations and camera transforms.
It reuses the existing meter-scale GraspNet models and converts object poses to the
table frame under `assets/scene/dexgraspnet2_train/scene_cfg`, with one target config
per object per scene. No RGB/depth images or grasp labels are extracted; each archive
is removed after a successful import unless `--keep-archives` is set. Per-archive
progress and resumable import records are kept alongside the scene configs. This
preparation does not launch grasp generation or clutter filtering.

Filter the reusable single-object library in those layouts with the independent
**4. GraspNet：官方场景 MuJoCo 抓取过滤** launch, or run
`python scripts/filter_graspnet_scene_grasps.py --scene scene_0000 --workers 84`.
Use `--scene 'scene_*'` for all training layouts. This CPU-only stage checks the
pregrasp, sampled approach path, scene/self collisions and target contact for all
33 grasp types without regenerating grasps or scenes. Add `--lift` for scene-level
dynamics validation. It preserves zero-grasp layouts, tracks progress and resumes
unchanged targets while detecting additions to the growing source library.
See [the scene filtering guide](docs/graspnet_scene_filter_zh.md) for outputs and thresholds.

For VS Code, install the **Dev Containers** extension and run **Dev Containers: Reopen in Container**. The checked-in `.devcontainer` configuration uses the same CUDA image and mounts the repository at `/workspace`.

MuJoCo generation remains available through `op=grasp` and
`scripts/generate_native_clutter_grasps.py`; physical validation runs on the CPU.

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful for your research, please consider citing:
```
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```
