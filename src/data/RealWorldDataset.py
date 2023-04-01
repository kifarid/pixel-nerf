import glob
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform.rotation import Rotation as R
# import dataloader
from torch.utils.data import DataLoader

from src.util import get_image_to_tensor_balanced, dict_to_tensor, repeat_interleave


def plot_coordinate_frame(image, cam_matrix, frame, length=0.1, thickness=2, wait=True):
    pose = cam_matrix @ frame
    # rescale the pose
    # Extract the rotation and translation from the 4x4 pose matrix
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # scale the axes and do a translation to get end points
    rotation = rotation * length + translation[:, None]

    start = translation[:2] / translation[2]
    end = rotation[:2] / rotation[2]
    # Convert the start and end points to integers
    x_start, y_start = start.astype(int)
    x_x_end, x_y_end = end[:, 0].astype(int)
    y_x_end, y_y_end = end[:, 1].astype(int)
    z_x_end, z_y_end = end[:, 2].astype(int)

    image = np.ascontiguousarray(image, dtype=np.uint8)
    # Plot the x-axis in red
    cv2.line(image, (x_start, y_start), (x_x_end, x_y_end), (0, 0, 255), thickness)

    # Plot the y-axis in green
    cv2.line(image, (x_start, y_start), (y_x_end, y_y_end), (0, 255, 0), thickness)

    # Plot the z-axis in blue
    cv2.line(image, (x_start, y_start), (z_x_end, z_y_end), (255, 0, 0), thickness)

    return image


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


def get_cam_matrix(f, c, static_extrinsics):
    """
    :param f: focal length (bs, n_views, 2)
    :param c: principal point (bs, n_views, 2)
    :param static_extrinsics: static extrinsics (bs, n_views, 4, 4)
    """
    # construct the intrinsic matrix out of focal length (bs, n_views, 2) and principal cx (bs, n_views) and cy (bs, n_views)
    if len(f.shape) == 2:
        f = f[None, ...]
        c = c[None, ...]
        static_extrinsics = static_extrinsics[None, ...]
    intrinsic_matrix = np.zeros((f.shape[0], f.shape[1], 3, 3))
    intrinsic_matrix[:, :, 0, 0] = f[:, :, 0]
    intrinsic_matrix[:, :, 1, 1] = f[:, :, 1]
    intrinsic_matrix[:, :, 0, 2] = c[:, :, 0]
    intrinsic_matrix[:, :, 1, 2] = c[:, :, 1]
    intrinsic_matrix[:, :, 2, 2] = 1

    intrinsic_full_matrix = np.zeros((f.shape[0], f.shape[1], 4, 4))
    intrinsic_full_matrix[:, :, :3, :3] = intrinsic_matrix
    # get the gripper pose in the static camera frame
    cam_matrix = intrinsic_full_matrix @ np.linalg.inv(static_extrinsics)
    return cam_matrix


def load_episode(episode):
    episode_filtered = []

    # loading the npz files for the corresponding episode
    for i, path in enumerate(episode):
        # skip in valid data
        try:
            episode_filtered.append(dict(np.load(str(path), allow_pickle=True)))
        except Exception as e:
            print(f"error {e} skipping invalid data {path}, {i}")
            continue
    
    #print(f"loaded {len(episode_filtered)} frames")

    if len(episode_filtered) == 0:
        return {}
    
    episode = {}
    keys = episode_filtered[0].keys()
    for k in keys:
        if k == 'robot_state':
            if k not in episode:
                episode[k] = {}

            sub_keys = episode_filtered[0][k][None][0].keys()
            for k2 in sub_keys:
                episode[k][k2] = np.array([d[k][None][0][k2] for d in episode_filtered])

        elif k == "action":
            if k not in episode:
                episode[k] = {}
            sub_keys = episode_filtered[0][k][None][0].keys()
            for k2 in sub_keys:
                if k2 == "motion":
                    if k2 not in episode[k]:
                        episode[k][k2] = {}
                    for i, k3 in enumerate(["pos", "quat", "close"]):
                        episode[k][k2][k3] = np.array([d[k][None][0][k2][i] for d in episode_filtered])
                else:
                    episode[k][k2] = np.array(
                        [d[k][None][0][k2] for d in episode_filtered if len(d[k][None][0][k2]) > 0])
        else:
            if len(episode_filtered[0][k].shape) > 0:
                if episode_filtered[0][k].shape[0] == 0:
                    continue
            episode[k] = np.array([d[k] for d in episode_filtered])

    return episode


class Episode(dict):
    def __init__(self, data, from_npz=True, filter_repeated=True):
        if from_npz:
            episode = load_episode(data)
            super().__init__(**episode)
        else:
            super().__init__(**data)

        # Set the keys of the episode as attributes for easier access

        # filter the repeated states
        if filter_repeated:
            self.filter_repeated()

        for key in self.keys():
            setattr(self, key, self[key])

    def filter_repeated(self):
        if "robot_state" not in self:
            return
        # concatentate the robot state's tcp_pos, tcp_orn, gripper_openining_width to a state
        state = np.concatenate((self["robot_state"]["tcp_pos"],
                                self["robot_state"]["tcp_orn"],
                                self["robot_state"]["gripper_opening_width"][:, None]), axis=1)

        # get the difference between the current and next state
        state_diff = np.linalg.norm(np.diff(state, axis=0), ord=np.inf, axis=1)
        # get first state with diff > 1e-5
        start_idx = np.argmax(state_diff > 1e-3)
        # get last state with diff > 1e-5
        end_idx = len(state_diff) - np.argmax(state_diff[::-1] > 1e-3)
        episode = self[start_idx:end_idx]
        for k, v in episode.items():
            self[k] = v

    def __len__(self):
        # get the first key
        key = list(self.keys())[0]
        return len(self[key])

    def __getitem__(self, index):

        if isinstance(index, str):
            return super().__getitem__(index)

        elif isinstance(index, int):
            # Create a new episode containing only the requested keys
            # index = max(0, min(index, len(self) - 1))  # clamp index to [0, len(self) - 1]
            sliced_episode = {}
            for k, v in self.items():
                if isinstance(v, dict):
                    v_episode = Episode(v, from_npz=False, filter_repeated=False)
                    sliced_episode[k] = v_episode[index]
                else:
                    sliced_episode[k] = v[index]
            return sliced_episode

        elif isinstance(index, slice):
            # the end index is always in the episode
            # TODO problem with negative indices specially stop at 0
            start = int(index.start) if index.start is not None else 0
            stop = int(index.stop) if index.stop is not None else None
            step = int(index.step) if index.step is not None else 1

            # Create a new episode containing only the requested keys
            sliced_episode = {}  # Episode()
            for k, v in self.items():
                if isinstance(v, dict):
                    v_episode = Episode(v, from_npz=False, filter_repeated=False)
                    sliced_episode[k] = v_episode[start:stop:step]
                else:

                    # sliced_episode[k] = v[max(0, start):stop:step]
                    # if sliced_episode[k].shape[0] < (stop - start) // step:
                    # sliced_episode[k] = np.pad(sliced_episode[k], (((stop - start) //step - sliced_episode[k].shape[0], 0), *((len(sliced_episode[k].shape)-1)*((0, 0),))), 'edge')
                    # sliced_episode[k] = np.pad(sliced_episode[k], ((), *((len(sliced_episode[k].shape)-1)*((0, 0),))), 'edge')

                    # get the first and last index
                    first_idx = start if start >= 0 else np.abs(int(start + np.ceil(np.abs(start) / step) * step))
                    first_idx = min(first_idx, len(self) - 1)
                    sliced_episode[k] = v[first_idx:stop:step]

                    if stop is not None:
                        length_slice = np.abs(int(np.ceil((stop - start) / step)))
                    else:
                        length_slice = int(np.ceil((len(self) - start) / step)) if step > 0 else start // step

                    if sliced_episode[k].shape[0] < length_slice:
                        # TODO test this
                        left_pad = length_slice - sliced_episode[k].shape[
                            0]  # 0 if start >= 0 else int(np.ceil(np.abs(start) // step))

                        sliced_episode[k] = np.pad(sliced_episode[k], (
                            (left_pad, 0), *((len(sliced_episode[k].shape) - 1) * ((0, 0),))), 'edge')

            sliced_episode = Episode(sliced_episode, from_npz=False, filter_repeated=False)
            return sliced_episode


        else:
            raise TypeError(f"Invalid index type {type(index)}")

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Set the key as an attribute for easier access
        setattr(self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)

        # Delete the attribute corresponding to the key
        if hasattr(self, key):
            delattr(self, key)


# create a data wrapper class fom the teleop_data
class TeleopData(torch.utils.data.Dataset):

    def __init__(self, directory, stage="train", z_near=0.01, z_far=2, frame_rate=1, frame_stack=1,
                 image_size=(150, 200), relative=True, action_from_next_state=False, world_scale=1.0, views=None,
                 image_scale=1.0, allow_reverse=True, state_keys=None, image_keys=None, t_views= False):

        self.directory = directory
        self.frame_stack = frame_stack
        self.allow_reverse = allow_reverse
        self.frame_rate = frame_rate
        self.relative = relative
        self.action_from_next_state = action_from_next_state
        self.frames_paths = []
        for root, dirs, files in os.walk(self.directory):
            # Collect file paths that match "frame_*" in the current subdirectory
            subdirectory_paths = sorted([os.path.join(root, f) for f in files if f.startswith('frame_')])
            if subdirectory_paths:
                # Add the list of file paths to the frames_paths list
                self.frames_paths.append(subdirectory_paths)
        self.frames_paths = sorted(self.frames_paths, key=lambda x: x[0])
        # if len(self.frames_paths) == 1:
        #     self.frames_paths = self.frames_paths[0]
        self.ep_start_end_paths = sorted(glob.glob(os.path.join(self.directory, "*", "ep_start_end_ids.npy")))

        self.ep_start_end_ids = None
        self.episodes = None
        self.cumulative_episode_lengths = None
        self.episode_lengths = None

        self.stage = stage
        self.image_size = image_size
        self.world_scale = world_scale

        self.views = views

        self.image_to_tensor = get_image_to_tensor_balanced()
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)  # (TODO check this)
        )

        self.z_near = z_near  # 0.1 TODO: check this according to workspace dimensions
        self.z_far = z_far  # 1.6
        self.lindisp = False
        print("Loading Teleop dataset", self.directory)

        # load the camera_info
        self.get_camera_info()
        # load the teleop_data
        self.get_teleop_data()

        self.state_keys = state_keys
        self.image_keys = image_keys
        self.t_views = t_views

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
        cam_info_dir = sorted(glob.glob(os.path.join(self.directory, "*", "camera_info.npz")))[0]
        self.cam_info = np.load(cam_info_dir, allow_pickle=True)
        # convert from files to a dict
        self.cam_info = dict(self.cam_info)
        # build the intrinisic matrix for all cameras
        if "static_intrinsics" in self.cam_info:
            static_intrinsic = self.cam_info["static_intrinsics"]
            self.cam_info["static_intrinsics"] = self.build_intrinsics(static_intrinsic[None][0])
            # create the camera matrix from the intrinsics

        # repeat for the dynamic camera
        if "gripper_intrinsics" in self.cam_info:
            dynamic_intrinsic = self.cam_info["gripper_intrinsics"]
            self.cam_info["gripper_intrinsics"] = self.build_intrinsics(dynamic_intrinsic[None][0])
        return self.cam_info

    def get_teleop_data(self):
        # load the ep_start_end_ids
        episodes = []
        for frames, ep_start_end_paths in zip(self.frames_paths, self.ep_start_end_paths):
            epi_start_end_ids = np.load(ep_start_end_paths, allow_pickle=True)
            sub_episodes = [frames[start:end] for start, end in epi_start_end_ids]
            for sub_episode in sub_episodes:
                episode_obj = Episode(sub_episode)
                # check if the episode obj is an empty dict
                if len(episode_obj.keys()) > 0:
                    episodes.append(episode_obj)

        self.episodes = episodes
        # stop here
        self.episode_lengths = [len(ep) for ep in self.episodes]
        self.cumulative_episode_lengths = np.cumsum(self.episode_lengths)

    def get_action_from_state(self, next_state, close):
        '''
        next_state (dict): next state with tcp_pos, tcp_orn  correspnding to t + frame_rate
        close (int): 0 or 1 from the current gripper
        '''
        action = {}
        action["pos"] = next_state["tcp_pos"]
        action["quat"] = next_state["tcp_orn"]
        action["close"] = close
        return action

    def get_relative_action(self, current_state, action):
        '''
        action (dict): action with pos, quat, close, correspnding to t = t + frame_rate - 1
        current_state (dict): current state with tcp_pos, tcp_orn  correspnding to t
        '''
        # get the relative action
        relative_action = {}
        relative_action["pos"] = action["pos"] - current_state["tcp_pos"]
        relative_action["quat"] = action["quat"] - current_state["tcp_orn"]
        relative_action["close"] = action["close"]

        return relative_action

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

    def process_action(self, action):
        # process the action,
        action = dict_to_tensor(action)
        return action

    def process_state(self, state):
        # process the state
        state = dict_to_tensor(state)
        return state

    def get_gripper_pose(self, gripper_pose):
        pass

    def process_sample(self, sample):
        # process the sample

        action_orig = self.process_action(sample['action']['motion'])

        ref = torch.tensor(sample["action"]["ref"] == 'abs', dtype=torch.bool)
        robot_state = self.process_state(sample['robot_state'])

        action = self.process_action(sample['sampled_action']['motion'])
        next_robot_state = self.process_state(sample['next_robot_state'])

        images = []
        depths = []
        for key in sorted(sample.keys()):
            if "rgb" in key:
                if self.image_keys is not None:
                    if key not in self.image_keys:
                        continue
                if len(sample[key].shape) < 3:
                    continue
                images.append(sample[key])

            if "depth" in key:
                if len(sample[key].shape) < 3:
                    continue
                depths.append(sample[key])

        images_shps = [img.shape[-3:-1] for img in images]  # (H, W)
        images_tensor = self.process_cams(images, depths)  # (N, T, C, H, W)
        cx, cy, fx, fy, poses = [], [], [], [], []

        # intialize the tcp pose as a nx4x4 matrix where n is the number of frames and each 4x4 matrix is identity
        tcp_pose = get_tcp_pose(sample["robot_state"]["tcp_pos"],
                                sample["robot_state"]["tcp_orn"])  # (T, 4, 4) or (4, 4)
        # getting gripper cam info
        if "gripper_intrinsics" in self.cam_info:
            intrin_gripper = self.cam_info["gripper_intrinsics"]
            pose_T_tcp_cam = self.cam_info["gripper_extrinsic_calibration"]
            gripper_poses = torch.tensor(tcp_pose @ pose_T_tcp_cam, dtype=torch.float32)  # (T, 4, 4) or (4, 4)
            gripper_poses = gripper_poses if self.frame_stack > 1 else gripper_poses.unsqueeze(0) 
            scale_x, scale_y = 1, 1
            if images[0].shape[-2:] != self.image_size:
                scale_y, scale_x = self.image_size[0] / images_shps[0][0], self.image_size[1] / \
                                images_shps[0][1]
            cx.append(intrin_gripper["cx"] * scale_x)
            cy.append(intrin_gripper["cy"] * scale_y)
            fx.append(intrin_gripper["fx"] * scale_x)
            fy.append(intrin_gripper["fy"] * scale_y)

        if "static_intrinsics" in self.cam_info:
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

            poses = torch.stack(poses, dim=0).unsqueeze(1).repeat(1, self.frame_stack, 1,
                                                              1)  # repeat for frame stack (N, T, 4, 4)
        

        if "gripper_intrinsics" in self.cam_info and "static_intrinsics" in self.cam_info:
            poses = torch.cat([gripper_poses.unsqueeze(0), poses])
        elif "gripper_intrinsics" in self.cam_info:
            poses = gripper_poses.unsqueeze(0)
        elif "static_intrinsics" in self.cam_info:
            poses = poses
        else:
            raise ValueError("No camera info found")

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

        if self.action_from_next_state:
            close = action_orig['close'] if action_orig['close'].shape[0] == 1 else action_orig['close'][-1]
            action = self.get_action_from_state(next_robot_state, close=close)

        if self.relative:
            # get last robot_state in case of frame stack
            robot_state_last_t = {}
            if self.frame_stack > 1:
                for k in robot_state.keys():
                    robot_state_last_t[k] = robot_state[k][-1]
            else:
                robot_state_last_t = robot_state

            action = self.get_relative_action(robot_state_last_t, action)

        if self.t_views:
            #  images from (N, T, C, H, W) to (NxT ,1, C, H, W) ordered as view1_t1, view1_t2, ..., view2_t1, view2_t2, ...
            #  poses from (N, T, 4, 4) to (NxT, 1, 4, 4) ordered as view1_t1, view1_t2, ..., view2_t1, view2_t2, ...
            images_tensor = images_tensor.reshape(-1, 1, *images_tensor.shape[2:])
            poses = poses.reshape(-1, 1, *poses.shape[2:])
            f = repeat_interleave(f, self.frame_stack) #f.repeat(self.frame_stack, 1)
            c = repeat_interleave(c, self.frame_stack) #c.repeat(self.frame_stack, 1)


        result = {
            "images": images_tensor,  # .squeeze(),
            "poses": poses@self._coord_trans,  # .squeeze(),
            "focal": f,  # .squeeze(),
            "c": c,  # .squeeze(),
            "ref": ref,
            "robot_state": robot_state,
            "action_orig": action_orig,
            "action": action,
            "image_size": torch.tensor(self.image_size, dtype=torch.float32),
            "world_scale": torch.tensor(self.world_scale, dtype=torch.float32),
        }
        return result

    def __len__(self):
        return int(self.cumulative_episode_lengths[-1])

    def __getitem__(self, index):
        ep_index = np.argmax(self.cumulative_episode_lengths > index)
        idx_in_ep = index - int(self.cumulative_episode_lengths[ep_index - 1]) if ep_index > 0 else index
        episode = self.episodes[ep_index]
        # TODO test this
        extra_frames = self.frame_stack - 1 if self.frame_stack > 1 else 0
        sample = episode[idx_in_ep] if extra_frames == 0 else episode[
                                                              idx_in_ep - self.frame_rate * extra_frames:idx_in_ep + 1:self.frame_rate]

        idx_next_sample = max(min(idx_in_ep + self.frame_rate, len(episode) - 1), 0)
        idx_action = max(min(idx_in_ep + self.frame_rate - 1, len(episode) - 1), 0)
        sample["next_robot_state"] = episode[idx_next_sample]["robot_state"]  # dict of vectors
        sample["sampled_action"] = episode[idx_action]["action"]  # dict of vectors
        result = self.process_sample(sample)
        result["ep_index"] = torch.tensor(ep_index, dtype=torch.int64)
        result["idx_in_ep"] = torch.tensor(idx_in_ep, dtype=torch.int64)
        # if self.frame_stack == 0 else episode[idx_in_ep + self.frame_rate:idx_in_ep + self.frame_rate +
        # self.frame_stack * self.frame_rate + 1:self.frame_rate]

        if self.allow_reverse:
            stop = idx_in_ep - 1 if idx_in_ep > 0 else None
            reverse_sample = episode[idx_in_ep] if extra_frames == 0 else episode[
                                                                          idx_in_ep + extra_frames * self.frame_rate: stop:-self.frame_rate]
            idx_next_sample = max(min(idx_in_ep - self.frame_rate, len(episode) - 1), 0)
            idx_action = max(min(idx_in_ep - self.frame_rate - 1, len(episode) - 1), 0)
            reverse_sample["next_robot_state"] = episode[idx_next_sample]["robot_state"]
            reverse_sample["sampled_action"] = episode[idx_action]["action"]

            # reverse_next_sample = episode[max(idx_in_ep - self.frame_rate, 0)]
            reverse_result = self.process_sample(reverse_sample)
            reverse_result = {f"reverse_{key}": value for key, value in reverse_result.items()}
            # next_result["next_idx_in_ep"] = torch.tensor(idx_in_ep - self.frame_rate, dtype=torch.int64)
            result.update(reverse_result)
            # if self.frame_stack == 0 else episode[
            # idx_in_ep + self.frame_rate:idx_in_ep +
            # self.frame_rate - self.frame_stack * self.frame_rate - 1:-self.frame_rate]
        return result


if __name__ == "__main__":
    frame_stack = 10
    dataset = TeleopData(
        "/Users/kfarid/Desktop/Education/MSc_Freiburg/research/robot_io/expert_data/train",
        frame_stack=2,
        frame_rate=3,
        views=[1, 2],
        relative=True,
        # image_size=(128, 128),
        # views_per_scene=3,
        # world_scale=0.1,

    )
    dataset[20]
    dataset[0]
    dataset[163]
    actions_max = []
    actions_min = []
    frames = []
    frames_2 = []
    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    for i, sample in enumerate(dataloader):
        print(i)
        print(list(sample.keys()))
        # get images
        images = sample["images"]
        actions = torch.cat([sample["action"]["pos"], sample["action"]["quat"]], axis=-1)
        act_max = torch.cat([sample["action"]["pos"], sample["action"]["quat"]], axis=-1).max(0)[0].max(0)[0]
        act_min = torch.cat([sample["action"]["pos"], sample["action"]["quat"]], axis=-1).min(0)[0].min(0)[0]
        actions_min.append(act_min)
        actions_max.append(act_max)
        print(images.shape)
        last_images = images[0, :, -1, ...]  # if frame_stack - 1 else images[0]
        last_images_all = torch.cat([last_images[j] for j in range(last_images.shape[0])], dim=1) if len(
            last_images.shape) == 4 else last_images
        frame = last_images_all.permute(1, 2, 0).numpy()
        # convert from -1,1 to 0,1 and uint8
        frame = (frame + 1) / 2 * 255
        frames.append(frame.astype(np.uint8))
        if frame_stack - 1:
            tcp_pos, tcp_orn = sample["robot_state"]["tcp_pos"][0, -1].numpy(), sample["robot_state"]["tcp_orn"][
                0, -1].numpy()
        else:
            tcp_pos, tcp_orn = sample["robot_state"]["tcp_pos"][-1].numpy(), sample["robot_state"]["tcp_orn"][
                -1].numpy()

        poses = sample["poses"].numpy()[:, :, -1, ...]
        tcp_pose = get_tcp_pose(tcp_pos=tcp_pos, tcp_orn=tcp_orn)

        cam_matrix = get_cam_matrix(sample["f"].numpy(), sample["c"].numpy(), poses)

        last_images_with_pose = []

        for j in range(0, last_images.shape[0]):
            img = (last_images[j].permute(1, 2, 0).numpy() + 1) / 2 * 255
            img = img.astype(np.uint8)
            last_images_with_pose.append(plot_coordinate_frame(img, cam_matrix[0, j], tcp_pose))
        last_images_all = np.concatenate(last_images_with_pose, axis=1)

        # frame_2 = np.transpose(last_images_all, (1, 2, 0))
        frame_2 = last_images_all.astype(np.uint8)
        frames_2.append(frame_2)

    # create a gif
    imageio.mimsave('test.gif', frames, fps=10)
    imageio.mimsave('test_2.gif', frames_2, fps=10)
    print("max", torch.stack(actions_max).max(0)[0])
    print("min", torch.stack(actions_min).min(0)[0])
