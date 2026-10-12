"""Lazy operation imports.

Several operations initialize CUDA libraries at import time. Loading only the
selected operation keeps CPU-only MuJoCo tools independent from Warp/CUDA.
"""

from importlib import import_module


_OP_MODULES = {
    "operate_tmpl": "convert_template",
    "operate_init": "gen_init",
    "operate_grasp": "gen_grasp",
    "operate_grasp_admm": "gen_grasp_admm",
    "operate_eval": "eval_data",
    "operate_stat": "compute_stats",
    "operate_vusd": "vis_usd",
    "operate_v3d": "vis_3d",
    "operate_traj": "gen_traj",
    "operate_clutter": "gen_clutter",
}

__all__ = list(_OP_MODULES)


def __getattr__(name: str):
    if name not in _OP_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_OP_MODULES[name]}"), name)
    globals()[name] = value
    return value
