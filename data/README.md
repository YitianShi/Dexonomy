---
license: cc-by-nc-4.0
task_categories:
- robotics
tags:
- grasping
- dexterous-manipulation
- computer-vision
size_categories:
- 1M<n<10M
---

# Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy

[Paper](https://huggingface.co/papers/2504.18829) | [Project page](https://pku-epic.github.io/Dexonomy/) | [Code](https://github.com/JYChen18/Dexonomy)

## Overview

This dataset supports generalizable dexterous grasping with suitable grasp types, a fundamental skill for intelligent robots. It presents an efficient pipeline capable of synthesizing **contact-rich, penetration-free, and physically plausible** dexterous grasps for any grasp type, object, and articulated hand, starting from just one human-annotated template per hand and grasp type.

Using this algorithm, the Dexonomy dataset was constructed, containing 10.7k objects and 9.5M grasps, covering 31 grasp types in the GRASP taxonomy. The dataset is crucial for developing and training robust, type-conditional generative models for dexterous manipulation.

## Dataset structure

```
|- DGN_5k_processed.tar.gz  # Our pre-proposed meshes and training splits.
|- DGN_5k_vision.tar.gz     # The single-view point clouds. Only used for training networks. 
|- objaverse_5k_processed.tar.gz       
|- objaverse_5k_vision.tar.gz       
|_ Dexonomy_GRASP_shadow.tar.gz # 9.5M successful grasps including 31 types from the GRASP taxonomy
```

## Citation

If you find this work useful for your research, please consider citing:
```bibtex
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```