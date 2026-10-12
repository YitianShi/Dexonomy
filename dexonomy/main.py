import sys
import random
import logging
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np

import dexonomy.op as operations

seed = 12


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    try:
        # Clutter generation is CPU-only; spawned workers also import this entrypoint.
        if cfg.op_name != "clutter":
            import torch

        np.random.seed(seed)
        if cfg.op_name != "clutter":
            torch.manual_seed(seed)
        random.seed(seed)
        # PyOpenGL emits harmless INFO messages in headless MuJoCo processes.
        logging.getLogger("OpenGL").setLevel(logging.WARNING)
        getattr(operations, f"operate_{cfg.op_name}")(cfg)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
        sys.exit(1)
    return


if __name__ == "__main__":
    main()
