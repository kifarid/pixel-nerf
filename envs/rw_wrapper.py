"""Image dataset."""

import os
import pickle
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from src.util import get_image_to_tensor_balanced, dict_to_tensor
from PIL import Image
import torch
import hydra
from omegaconf import OmegaConf
import torch.nn.functional as F
from collections import defaultdict, OrderedDict

from collections import deque
from typing import Any, NamedTuple
import gym
import numpy as np

def get_tcp_pose(tcp_pos, tcp_orn):
    # create gripper pose matrix
    tcp_rot = R.from_quat(tcp_orn).as_matrix()

    if len(tcp_pos.shape) > 1:
        tcp_pose = np.eye(4)[None, ...].repeat(tcp_pos.shape[0], axis=0)
        tcp_pose[:, :3, :3] = tcp_rot
        tcp_pose[:, :3, 3] = tcp_pos
    else:
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = tcp_rot
        tcp_pose[:3, 3] = tcp_pos

    return tcp_pose


class FrameStackIlWrapper(gym.Env):

    def __init__(self, env, num_frames, relative=True, action_mode="abs"):
        """
        Wrapper for stacking frames
        :param env: environment to wrap
        :param num_frames: number of frames to stack
        :param relative: if True, actions are relative to the current pose
        :param action_mode: "abs" or "rel" which action mode to use

        """

        self.cam_info = None
        self._env = env
        self._num_frames = num_frames
        self.relative == relative
        self.action_mode = action_mode

        self._frames = deque([], maxlen=num_frames)
        self._poses = deque([], maxlen=num_frames)

        self.get_camera_info()

    def process_intrinsics(self, intrinsics):

        if intrinsics["crop_coords"] is not None:
            intrinsics["cx"] -= (intrinsics["crop_coords"][2])  # x
            intrinsics["cy"] -= (intrinsics["crop_coords"][0])  # y
        if intrinsics["resize_resolution"] is not None:
            width = intrinsics["width"] if intrinsics["crop_coords"] is None else intrinsics["crop_coords"][
                                                                                      3] - \
                                                                                  intrinsics["crop_coords"][
                                                                                      2]
            height = intrinsics["height"] if intrinsics["crop_coords"] is None else \
                intrinsics["crop_coords"][1] - intrinsics["crop_coords"][0]
            s_x = intrinsics["resize_resolution"][0] / width
            s_y = intrinsics["resize_resolution"][1] / height
            intrinsics["fx"] *= s_x
            intrinsics["fy"] *= s_y
            intrinsics["cx"] *= s_x
            intrinsics["cy"] *= s_y
            # update the width and height

        intrinsics["camera_matrix"] = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                                                [0, intrinsics["fy"], intrinsics["cy"]],
                                                [0, 0, 1]])

        # add the projection matrix from camera matrix
        intrinsics["projection_matrix"] = np.concatenate((intrinsics["camera_matrix"], np.zeros((3, 1))),
                                                         axis=1)

        return intrinsics

    def build_intrinsics(self, intrinsics):
        # build the intrinisic matrix for all camera
        # create the camera matrix from the intrinsics

        if isinstance(intrinsics, dict):
            intrinsics = self.process_intrinsics(intrinsics)
        else:
            for i, cam_intrinsics in enumerate(intrinsics):
                cam_intrinsics = self.process_intrinsics(cam_intrinsics)
                intrinsics[i] = cam_intrinsics

        return intrinsics

    def get_camera_info(self):
        # load the camera_info
        camera_info = {}
        if self._env.camera_manager.gripper_cam is not None:
            camera_info["gripper_extrinsic_calibration"] = self._env.camera_manager.gripper_cam.get_extrinsic_calibration(self._env.camera_manager.robot_name)
            camera_info["gripper_intrinsics"] = self._env.camera_manager.gripper_cam.get_intrinsics()
        if self._env.camera_manager.static_cam is not None:
            if self._env.camera_manager.static_cam_count > 1:
                camera_info["static_extrinsic_calibration"] = tuple()
                camera_info["static_intrinsics"] = tuple()
                for cam in self._env.camera_manager.static_cam:
                    camera_info["static_extrinsic_calibration"] += (cam.get_extrinsic_calibration(self._env.camera_manager.robot_name),)
                    camera_info["static_intrinsics"] += (cam.get_intrinsics(),)
            else:
                camera_info["static_extrinsic_calibration"] = self._env.camera_manager.static_cam.get_extrinsic_calibration(self._env.camera_manager.robot_name)
                camera_info["static_intrinsics"] = self._env.camera_manager.static_cam.get_intrinsics()

        self.cam_info = camera_info

        # build the intrinisic matrix for all cameras
        static_intrinsic = self.cam_info["static_intrinsics"]
        self.cam_info["static_intrinsics"] = self.build_intrinsics(static_intrinsic[None][0])
        # create the camera matrix from the intrinsics

        # repeat for the dynamic camera
        dynamic_intrinsic = self.cam_info["gripper_intrinsics"]
        self.cam_info["gripper_intrinsics"] = self.build_intrinsics(dynamic_intrinsic[None][0])
        return self.cam_info

    def _transform_observation(self, result):
        assert len(self._frames) == self._num_frames
        assert len(self._poses) == self._num_frames

        obs = torch.cat(list(self._frames), axis=1) # (NV, T, C, H, W)
        poses = torch.cat(list(self._poses), axis=1) # (NV, T, 4, 4)

        result["images"] = obs
        result["poses"] = poses

        return self.add_batch_dim(result)

    def reset(self):
        obs = self._env.reset()
        obs = self.process_obs(obs)
        for _ in range(self._num_frames):
            self._frames.append(obs["images"])
            self._poses.append(obs["poses"])

        return self._transform_observation(obs)

    def get_abs_action(self, current_state, action):
        '''
        action (dict): action with pos, quat, close relative
        current_state (dict): current state with tcp_pos, tcp_orn  correspnding to t
        '''
        # get the relative action
        abs_action = {}
        abs_action["pos"] = action["pos"] + current_state["tcp_pos"]
        abs_action["quat"] = action["quat"] + current_state["tcp_orn"]
        abs_action["close"] = action["close"]

        return abs_action

    def get_rel_action(self, current_state, action):
        '''
        action (dict): action with pos, quat, close abs
        current_state (dict): current state with tcp_pos, tcp_orn  correspnding to t
        '''
        # get the relative action
        relative_action = {}
        relative_action["pos"] = action["pos"] - current_state["tcp_pos"]
        relative_action["quat"] = action["quat"] - current_state["tcp_orn"]
        relative_action["close"] = action["close"]

        return relative_action

    def add_batch_dim(self, obs):
        for key, value in obs.items():
            if isinstance(value, dict):
                obs[key] = self.add_batch_dim(value)
            elif type(obs[key]) == torch.Tensor:
                obs[key] = obs[key].unsqueeze(0)
            else:
                raise NotImplementedError("Unsupported type: {}".format(type(obs[key])))

        return obs

    def process_action(self, action):
        """
        process the action to pass to the environment from tensors to numpy
        action (
        """

        action_processed = {}
        action_processed["ref"] = self.action_mode

        for key in action.keys():
                action_processed[key] = action[key].squeeze().detach().cpu().numpy() if key in action else 0

        # check if relative is true then add current pose

        if self.relative and self.action_mode == "abs":
            action_processed=self.get_abs_action(self._env.get_obs()["robot_state"], action_processed)

        elif not self.relative and self.action_mode == "rel":
            action_processed=self.get_rel_action(self._env.get_obs()["robot_state"], action_processed)

        return action_processed

    def process_cams(self, images, depths=None):
        """
        takes in a list of images and returns a torch tensor
        """
        images_tensor = []
        depths_tensor = []  # TODO: add depth
        for image in images:
            if len(image.shape) == 3:
                # add the time dimension
                image = image[None]
            image_t = []
            for img in image:
                img_t = self.image_to_tensor(img)
                image_t.append(img_t)
            image_t = torch.stack(image_t, dim=0)  # (T, C, H, W)
            if image_t.shape[-2:] != self.image_size:
                image_t = F.interpolate(image_t, size=self.image_size, mode='area')

            images_tensor.append(image_t)

        images_tensor = torch.stack(images_tensor, dim=0)  # (N, T, C, H, W)
        return images_tensor

    def process_state(self, state):
        # process the state
        state = dict_to_tensor(state)
        return state

    def process_obs(self, obs):
        # process the sample

        robot_state = self.process_state(obs['robot_state'])

        images = []
        depths = []
        for key in sorted(obs.keys()):
            if "rgb" in key:
                if self.image_keys is not None:
                    if key not in self.image_keys:
                        continue
                if len(obs[key].shape) < 3:
                    continue
                images.append(obs[key])

            if "depth" in key:
                if len(obs[key].shape) < 3:
                    continue
                depths.append(obs[key])

        images_shps = [img.shape[-3:-1] for img in images]  # (H, W)
        images_tensor = self.process_cams(images, depths)  # (N, T, C, H, W)
        cx, cy, fx, fy, poses = [], [], [], [], []

        # intialize the tcp pose as a nx4x4 matrix where n is the number of frames and each 4x4 matrix is identity
        tcp_pose = get_tcp_pose(obs["robot_state"]["tcp_pos"],
                                obs["robot_state"]["tcp_orn"])  # (T, 4, 4) or (4, 4)
        # getting gripper cam info
        intrin_gripper = self.cam_info["gripper_intrinsics"]
        pose_T_tcp_cam = self.cam_info["gripper_extrinsic_calibration"]
        gripper_poses = torch.tensor(tcp_pose @ pose_T_tcp_cam, dtype=torch.float32)  # (T, 4, 4) or (4, 4)

        scale_x, scale_y = 1, 1
        if images[0].shape[-2:] != self.image_size:
            scale_y, scale_x = self.image_size[0] / images_shps[0][0], self.image_size[1] / \
                               images_shps[0][1]
        cx.append(intrin_gripper["cx"] * scale_x)
        cy.append(intrin_gripper["cy"] * scale_y)
        fx.append(intrin_gripper["fx"] * scale_x)
        fy.append(intrin_gripper["fy"] * scale_y)

        for i, (intrin, pose) in enumerate(
                zip(self.cam_info["static_intrinsics"], self.cam_info["static_extrinsic_calibration"]), start=1):
            scale_x, scale_y = 1, 1
            if images[i].shape[-2:] != self.image_size:
                scale_x, scale_y = self.image_size[0] / images_shps[i][0], self.image_size[1] / \
                                   images_shps[i][1]
            cx.append(intrin["cx"] * scale_x)
            cy.append(intrin["cy"] * scale_y)
            fx.append(intrin["fx"] * scale_x)
            fy.append(intrin["fy"] * scale_y)
            poses.append(torch.tensor(pose, dtype=torch.float32))

        poses = torch.stack(poses, dim=0).unsqueeze(1)
        gripper_poses = gripper_poses.unsqueeze(0)  # (T, 4, 4)
        poses = torch.cat([gripper_poses.unsqueeze(0), poses])

        f = torch.tensor([fx, fy], dtype=torch.float32).T
        c = torch.tensor([cx, cy], dtype=torch.float32).T

        if self.world_scale != 1.0:
            f *= self.world_scale
            poses[..., :3, 3] *= self.world_scale

        if self.views is not None:
            images_tensor = images_tensor[self.views]
            poses = poses[self.views]
            f = f[self.views]
            c = c[self.views]

        result = {
            "images": images_tensor,
            "poses": poses,
            "f": f,
            "c": c,
            "robot_state": robot_state,
            "image_size": torch.tensor(self.image_size, dtype=torch.float32),
            "world_scale": torch.tensor(self.world_scale, dtype=torch.float32),
        }
        return result

    def step(self, action):
        action = self.process_action(action)
        next_obs, _, _, _ = self._env.step(action)
        next_obs = self.process_obs(next_obs)

        self._frames.append(next_obs["images"])
        self._poses.append(next_obs["poses"])

        return self._transform_observation(next_obs)
