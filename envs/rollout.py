from collections import defaultdict
from functools import partial, reduce
import logging
from operator import add
from typing import Any, Dict, List, Tuple, Optional

from envs.rollout_video import RolloutVideo
import hydra
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import torch.distributed as dist

log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def get_video_tag(task):
    return f"_{task}"


class Rollout(Callback):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        env_cfg,
        skip_epochs,
        rollout_freq,
        video,
        num_rollouts,
        ep_len,
        empty_cache,
        log_video_to_file,
        save_dir: Optional[str] = None,
        id_selection_strategy="select_first",
    ):
        self.env = None  # type: Any
        self.env_cfg = env_cfg
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.video = video
        self.num_rollouts = num_rollouts
        self.ep_len = ep_len
        self.empty_cache = empty_cache
        self.log_video_to_file = log_video_to_file
        self.save_dir = save_dir
        self.rollout_video = None  # type: Any
        self.device = None  # type: Any
        self.num_solved = []

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        self.save_dir = pl_module.logger.save_dir if self.save_dir is None else self.save_dir
        if self.env is None:

            self.device = pl_module.device

            self.env = hydra.utils.instantiate(self.env_cfg)
            if self.video:
                self.rollout_video = RolloutVideo(
                    logger=pl_module.logger,
                    empty_cache=self.empty_cache,
                    log_to_file=self.log_video_to_file,
                    save_dir=self.save_dir,
                )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if pl_module.current_epoch >= self.skip_epochs and (pl_module.current_epoch + 1) % self.rollout_freq == 0:

            self.num_solved.append(self.env_rollouts(pl_module))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:  # type: ignore
        # TODO: remove lightning fixes callback hook

        if pl_module.current_epoch == 0:
            pl_module.log("average_sr", torch.tensor(0.0), on_step=False, sync_dist=True)

        elif pl_module.current_epoch >= self.skip_epochs and (pl_module.current_epoch + 1) % self.rollout_freq == 0:
            # after first validation epoch, create task lookup dictionaries

            if self.video:
                # log rollout videos
                self.rollout_video.log(pl_module.global_step)
            # collect the task rollout counters of all validation batches and sum across tasks

            success_rate = sum(self.num_solved) / (len(self.num_solved)*self.num_rollouts) * 100

            # log to cmd line
            log_rank_0(f"{success_rate:.0f}%"

            )
            print()
            pl_module.log(
                "average_sr",
                torch.tensor(success_rate),
                on_step=False,
                sync_dist=True,
            )
            self.num_solved = []

    def env_rollouts(
        self,
        pl_module: LightningModule,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: tuple(
               val_obs: Tensor,
               val_rgbs: tuple(Tensor, ),
               val_depths: tuple(Tensor, ),
               val_acts: Tensor,
               val_lang: Tensor,
               info: Dict,
               idx: int
            pl_module: LightningModule
        Returns:
            rollout_task_counter: tensor counting the number of successful tasks rollouts in this batch
        """

        counter = 0

        for _ in range(self.num_rollouts):
            episode, total_reward = [], 0
            self.env.set_task(self.env.task)
            obs = self.env.reset()
            info = self.env.get_info()
            reward = 0
            # print(f'maxstep is {max_steps}')
            record_video = self.video
            if record_video:
                self.rollout_video.new_video(tag=get_video_tag(f"ravens_task"))
            success = False

            if record_video:
                # update video
                # squeeze first dim in obs
                img_v1 = obs["images"][:, 1, ...].unsqueeze(1)
                self.rollout_video.update(img_v1)

            for _ in range(self.ep_len):
                # print(f'info in start of episode: {info["d_poses"] if info is not None else None}')

                act = pl_module.step(obs)
                obs, reward, done, info = self.env.step(act)
                # print(f'info after_step start of episode: {info["d_poses"]}')
                if record_video:
                    # update video
                    # squeeze first dim in obs
                    img_v1 = obs["images"][:, 1, ...].unsqueeze(1)
                    self.rollout_video.update(img_v1)
                total_reward += reward
                #print(f'Total Reward: {total_reward} Done: {done}')
                if done:
                    break

            if total_reward > 0.99:
                success = True
                counter += 1

            if record_video:
                self.rollout_video.draw_outcome(success)
                self.rollout_video.write_to_tmp()

        return counter

