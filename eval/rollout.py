from collections import defaultdict
from functools import partial, reduce
import logging
from typing import Any, Dict, List, Tuple, Optional

import hydra

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch

import os
import sys
#arg parser
import argparse

#omegaconf
from omegaconf import DictConfig, OmegaConf


log_print = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append('/home/faridk/pixel-nerf')
sys.path.append('/home/faridk/pixel-nerf/src')
from src.model.vanilla_bc_rw import Vanilla_BC_RW
from util import instantiate_from_config


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.RealWorldDataset import TeleopData
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


@hydra.main(config_path="../configs/bc/real_world", config_name="vanilla_rollout")
def main(cfg: Dict[str, Any]) -> None:
    # set up logging
    model = instantiate_from_config(cfg.model)
    model.eval()
    dataset = TeleopData(
    "/work/dlclarge2/faridk-nerf_il/data/expert_data/val",
    frame_stack=3,
    frame_rate=2,
    views=[1, 2],
    relative=True,
    # image_size=(128, 128),
    # views_per_scene=3,
    # world_scale=0.1,

    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    for i, sample in enumerate(dataloader):
        #predict actions
        preds = model.step(sample)
        print("predicited positions: \n", preds["pos"], " \n actual positions: \n",sample["action"]["pos"])
        #print the difefrence between the predicted and the ground truth
        print("diff is:", preds["pos"]-sample["action"]["pos"])

if __name__ == "__main__":
    main()