import os
import argparse
from argparse import ArgumentParser
from space.config import cfg


def get_config():
    # parser = ArgumentParser()
    # parser.add_argument("--task", type=str, default="api", metavar="TASK", help="What to do. See engine")
    # parser.add_argument(
    #     "--config-file",
    #     type=str,
    #     default="space/configs/atari_spaceinvaders.yaml",
    #     metavar="FILE",
    #     help="Path to config file",
    # )

    # parser.add_argument(
    #     "opts", help="Modify config options using the command line", default=None, nargs=argparse.REMAINDER
    # )
    # args = parser.parse_args()
    config_file = "space/configs/atari_spaceinvaders.yaml"
    if config_file:
        cfg.merge_from_file(config_file)
    # if args.opts:
    #     cfg.merge_from_list(args.opts)

    # Use config file name as the default experiment name
    if cfg.exp_name == "":
        if config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(config_file))[0]
        else:
            raise ValueError("exp_name cannot be empty without specifying a config file")

    # Seed
    import torch

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np

    np.random.seed(cfg.seed)

    return cfg, "api"
