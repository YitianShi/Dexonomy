# Dexonomy

[Project Page](https://pku-epic.github.io/Dexonomy) | [Paper](https://arxiv.org/abs/2504.18829) | [Dataset](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy/tree/main) | [Learning](https://github.com/JYChen18/DexLearn)

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

The Docker runtime supports the repository's official `v3d` (OBJ) and `vusd` (USD) visualization operations:

```bash
# Unpack the five core archives into the README-compatible paths.
python scripts/prepare_data.py --group core
docker compose build dexonomy

# Export official OBJ or USD visualization files for one taxonomy.
docker compose run --rm dexonomy dexrun op=v3d hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_3d_dir=output/official_visualization/v3d
docker compose run --rm dexonomy dexrun op=vusd hand=shadow \
  op.data=succ_grasp op.n_max=5 n_worker=4 tmpl_name=1_Large_Diameter \
  succ_grasp_dir=data/extracted/Dexonomy_GRASP_shadow/succ_collect \
  vis_usd_dir=output/official_visualization/vusd
```

The outputs are written to `output/official_visualization/`. See [docs/docker_visualization_zh.md](docs/docker_visualization_zh.md) for data mapping, direct `dexrun` commands, `usdview` usage, and GPU synthesis commands.

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
