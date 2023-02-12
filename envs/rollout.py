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
import pandas as pd
import os

log_print = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

        pl_module.eval_all_models()
        # get preds
        self.preds = pl_module.forward(batch)
        self.acts_data = batch["actions"]
        self.batch = batch
        if pl_module.current_epoch >= self.skip_epochs and ((pl_module.current_epoch + 1) % self.rollout_freq == 0) and (batch_idx) % self.rollout_freq == 0:

            self.num_solved.append(self.env_rollouts(pl_module))
        pl_module.eval_all_models()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:  # type: ignore
        # TODO: remove lightning fixes callback hook
        pl_module.eval_all_models()
        if pl_module.current_epoch == 0:
            pl_module.log("average_sr", torch.tensor(0.0), on_step=False, sync_dist=True)

        elif pl_module.current_epoch >= self.skip_epochs and (pl_module.current_epoch + 1) % self.rollout_freq == 0:
            # after first validation epoch, create task lookup dictionaries

            if self.video:
                # log rollout videos
                self.rollout_video.write_to_tmp()
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
        pl_module.eval_all_models()

    # def save_images(self, obs, t):

    #     fig = plt.figure()
    #     plt.imshow(obs["images"][0, 0].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_obs_0_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
        
    #     fig = plt.figure()
    #     plt.imshow(obs["images"][0, 1].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_obs_1_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
        
    #     fig = plt.figure()
    #     plt.imshow(obs["images"][0, 2].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_obs_2_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)

    #     fig = plt.figure()
    #     plt.imshow(self.batch["images"][t, 0].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_batch_0_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)

    #     fig = plt.figure()
    #     plt.imshow(self.batch["images"][t, 1].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_batch_1_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)

    #     fig = plt.figure()
    #     plt.imshow(self.batch["images"][t, 2].permute(1, 2, 0).detach().cpu().numpy())
    #     plt.axis('off')
    #     plt.savefig(f"episodes/images_from_batch_2_{t}.png", bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)


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
        # for debugging reasons only
        seed = -2
        
        # np.random.seed(seed)

        # make directory to store the episode data
        # os.makedirs("episodes", exist_ok=True)
        # true_acts = []
        # pred_acts = []
        # pred_batch_acts = []
        true_batch_acts = []
        # ee_poses = []
        # act_res = []
        # diffs = []

        for rl in range(self.num_rollouts):
            # # for debugging reasons only
            # #seed+= 2
            # np.random.seed(0) #TODO remove in case of more than 1
            episode, total_reward = [], 0
            self.env.set_task(self.env.task)
            #agent = self.env.env.task.oracle(self.env.env, steps_per_seg=1)
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
                img_v1 = obs["images"][:, 0, ...].unsqueeze(1)
                self.rollout_video.update(img_v1)

            #true_action = agent.act(obs, info)
            for t in range(self.ep_len):
                # print(f'info in start of episode: {info["d_poses"] if info is not None else None}')
                
                #true_action = agent.act(obs, info)
                act = pl_module.step(obs)
                
                # true_acts.append(true_action)
                # pred_acts.append(act)
                # ee_pose = self.env.get_ee_pose()
                # ee_poses.append(ee_pose)
                # act_res.append({'move_cmd_0': act["move_cmd_0"].detach().cpu().numpy() + ee_pose[0], 'move_cmd_1': act["move_cmd_1"].detach().cpu().numpy() + ee_pose[1]})
                # pred_batch_acts.append({'move_cmd_0':self.preds['move_cmd_0'][t], 'move_cmd_1': self.acts_data['move_cmd_1'][t], 'suction_cmd':self.acts_data['suction_cmd'][t]})
                true_batch_act = {'move_cmd_0':self.acts_data['move_cmd_0'][min(t, 6)].unsqueeze(0), 'move_cmd_1': self.acts_data['move_cmd_1'][min(t, 6)].unsqueeze(0), 'suction_cmd':self.acts_data['suction_cmd'][min(t, 6)].unsqueeze(0)}
                true_batch_acts.append(true_batch_act)
                #set everything other than move_cmd_0 in act to true_batch_act
                act_to_env = {}
                for key in true_batch_act.keys():
                    act_to_env[key] = act[key]
                    #if key == 'succion_cmd':
                    #    act_to_env[key] = true_batch_act[key]
                    # if key != 'move_cmd_0':
                    #     act_to_env[key] = true_batch_act[key]
                    # else:
                    #     act_to_env[key] = act[key]

                # #act["move_cmd_0"] = torch.tensor(true_action["move_cmd"][0]).unsqueeze(0).to(self.device)
                #act["move_cmd_1"] = torch.tensor(true_batch_acts["move_cmd"][1]).unsqueeze(0).to(self.device)
                #act["suction_cmd"] = torch.tensor(true_batch_acts["suction_cmd"]).unsqueeze(0).to(self.device)
                # #print(act, true_action, "act is")

                # #process action predicted and computed difference in move_cmd 0
                # processed_actions = self.env.process_action(act)
                # #get the norm distance between the true action and the predicted action key move_cmd
                # curr_diff = np.linalg.norm(true_action['move_cmd'][0] - np.array(processed_actions['move_cmd'][0]))
                # diffs.append(curr_diff)

                # # pred_action = {key: self.preds[key][t:t+1] for key in act.keys()}
                # pred_action["suction_cmd"] = torch.argmax(pred_action["suction_cmd"], dim=-1)
                

                obs, reward, done, info = self.env.step(act_to_env)
                # print(f'info after_step start of episode: {info["d_poses"]}')
                if record_video:
                    # update video
                    # squeeze first dim in obs
                    img_v1 = obs["images"][:, 0, ...].unsqueeze(1)
                    self.rollout_video.update(img_v1)
                total_reward += reward
                #print(f'Total Reward: {total_reward} Done: {done}')
                
            if total_reward > 0.99:
                success = True
                counter += 1
            
            # make a dataframe with acts, true acts, true batch acts, pred batch acts, diffs as columns and save them
            #df = pd.DataFrame({'true_acts': true_acts, 'pred_acts': pred_acts, 'true_batch_acts': true_batch_acts, 'pred_batch_acts': pred_batch_acts, 'ee_pose': ee_poses, 'act_res': act_res})
            #df.to_csv(f"episodes/episode.csv")
            

            if record_video:
                self.rollout_video.draw_outcome(success)
                #self.rollout_video.write_to_tmp()

        return counter

