"""Load full hand simulation dependencies only when an environment is requested."""

from importlib import import_module

from .basic import HandCfg

__all__ = [
    "HandCfg",
    "MuJoCo_OptEnv",
    "MuJoCo_EvalEnv",
    "MuJoCo_VisEnv",
    "MuJoCo_OptCfg",
    "MuJoCo_EvalCfg",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.mujoco_env"), name)
    globals()[name] = value
    return value
