"""Image dataset."""

import os
import pickle
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
import tensorflow as tf
from PIL import Image
import torch
import hydra
from omegaconf import OmegaConf
import torch.nn.functional as F
from ravens import tasks
from collections import defaultdict, OrderedDict

import gym
import numpy as np

import pkgutil
import sys
import tempfile
import time

from typing import Dict, Tuple, List, Optional, Union, Any

from ravens.tasks.grippers import Spatula
from ravens.utils import pybullet_utils
from ravens.utils import utils
from src.util.util import pose_from_config

from src.util.util import get_image_to_tensor_balanced, get_mask_to_tensor, action_dict_to_tensor


import pybullet as p

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # camera to world
    # Set default camera poses.
    front_position = (1., 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    invViewMatrixFront = np.eye(4)
    invViewMatrixFront[:3, :3] = R.from_quat(front_rotation).as_matrix()
    invViewMatrixFront[:3, 3] = front_position
    invViewMatrixFront[:3, 0] /= 1.1
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)
    # Default camera configs.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': front_position,
        'rotation': front_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': left_position,
        'rotation': left_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }, {
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': right_position,
        'rotation': right_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }
    ]

    front_position = (0.8, 0, 0.5)
    front_rotation = (np.pi / 5, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    invViewMatrixFront = np.eye(4)
    invViewMatrixFront[:3, :3] = R.from_quat(front_rotation).as_matrix()
    invViewMatrixFront[:3, 3] = front_position
    invViewMatrixFront[:3, 0] /= 1.1
    extra_views_configs = []
    extra_views = 10
    base = 90. / (extra_views // 2) / 2
    angle = base

    for _ in range(extra_views // 2):
        left_rotation = np.eye(4)
        left_rotation[:3, :3] = r = R.from_euler('z', -angle, degrees=True).as_matrix()
        invViewMatrixLeft = left_rotation.dot(invViewMatrixFront)  # viewMatrix.dot(left_rotation)
        r = R.from_matrix(invViewMatrixLeft[:3, :3])
        left_position = invViewMatrixLeft[:3, 3]
        left_rotation = r.as_quat()

        right_rotation = np.eye(4)
        right_rotation[:3, :3] = r = R.from_euler('z', angle, degrees=True).as_matrix()
        InvViewMatrixRight = right_rotation.dot(invViewMatrixFront)  # viewMatrix.dot(left_rotation)
        r = R.from_matrix(InvViewMatrixRight[:3, :3])
        right_position = InvViewMatrixRight[:3, 3]
        right_rotation = r.as_quat()

        extra_views_configs.extend([{
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': left_position,
            'rotation': left_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }, {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': right_position,
            'rotation': right_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }])

        angle += base

    CONFIG.extend(extra_views_configs)


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 n_agent_cams=3,
                 n_ee_cams=1,
                 use_egl=False):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.
          use_egl: Whether to use EGL rendering. Only supported on Linux. Should get
            a significant speedup in rendering when using.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        if use_egl and disp:
            raise ValueError('EGL rendering cannot be used with `disp=True`.')

        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = copy.deepcopy(RealSenseD415.CONFIG)[:n_agent_cams]
        self.ee_cams = copy.deepcopy(RealSenseD415.CONFIG)[:n_ee_cams]

        self.assets_root = assets_root

        agent_cams = self.get_space_from_camera_config(self.agent_cams)
        cam_spaces = self.get_space_from_camera_config(self.ee_cams)

        # extend each element ee_space with agent cams fields to get full observation space
        [ee.extend(agent) for ee, agent in zip(cam_spaces, agent_cams)]


        obs_space_dict = defaultdict(tuple)
        for i, _ in enumerate(cam_spaces[0]):
            # loop over available camera types (e.g. RealSenseD415)
            obs_space_dict['color'] += (cam_spaces[0][i],)
            obs_space_dict['depth'] += (cam_spaces[1][i],)
            obs_space_dict['segm'] += (cam_spaces[2][i],)

        obs_space_dict['color'] = gym.spaces.Tuple(obs_space_dict['color'])
        obs_space_dict['depth'] = gym.spaces.Tuple(obs_space_dict['depth'])
        obs_space_dict['segm'] = gym.spaces.Tuple(obs_space_dict['segm'])

        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)

        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        # Start PyBullet.
        disp_option = p.DIRECT
        if disp:
            disp_option = p.GUI
            if shared_memory:
                disp_option = p.SHARED_MEMORY
        client = p.connect(disp_option,
                           options='--background_color_red=0.0 --background_color_green=0.0 --background_color_blue=0.0')
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        self._egl_plugin = None
        if use_egl:
            assert sys.platform == 'linux', ('EGL rendering is only supported on '
                                             'Linux.')
            egl = pkgutil.get_loader('eglRenderer')
            if egl:
                self._egl_plugin = p.loadPlugin(egl.get_filename(),
                                                '_eglRendererPlugin')
            else:
                self._egl_plugin = p.loadPlugin('eglRendererPlugin')
            print('EGL renderering enabled.')

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        if disp:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        if task:
            self.set_task(task)

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
             for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def get_space_from_camera_config(self, camera_config):
        """Get space from camera config.

        Args:
          camera_config: A dict of camera configuration.

        Returns:
          A gym.spaces.Box object.
        """
        color_tuple = [gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
             for config in camera_config]

        depth_tuple = [gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
             for config in camera_config]

        segm_tuple = [gym.spaces.Box(-2.0, 20.0, config['image_size'], dtype=np.float32)
             for config in camera_config]

        return color_tuple, depth_tuple, segm_tuple

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)
        self.obj_ids[category].append(obj_id)
        return obj_id

    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH),
                                 [0, 0, -0.001])
        # pybullet_utils.load_urdf(
        #    p, os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, UR5_URDF_PATH))
        self.ee = self.task.ee(self.assets_root, self.ur5, 9, self.obj_ids)
        self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Reset end effector.
        self.ee.release()

        # Reset task.
        self.task.reset(self)
        self.update_ee_cam()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs, _, _, _ = self.step()
        return obs

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            timeout = self.task.primitive(self.movej, self.movep, self.ee, **action)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                self.update_ee_cam()
                obs = self._get_obs()
                return obs, 0.0, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            p.stepSimulation()

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        # Add ground truth robot state into info.
        self.update_ee_cam()
        info.update(self.info)
        obs = self._get_obs()

        return obs, reward, done, info

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        p.disconnect()

    def render(self, mode='rgb_array'):
        # Render only the color image from the first camera.
        # Only support rgb_array for now.
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0][0])
        return color

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        # print(config['position'], lookat, updir)
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (config['image_size'][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # Note when use_egl is toggled, this option will not actually use openGL
            # but EGL instead.
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0], config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config['image_size']))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)
        return color, depth, segm

    @property
    def info(self):
        """Environment info variable with camera poses, object poses, dimensions, and colors."""

        # Some tasks create and remove zones, so ignore those IDs.
        # removed_ids = []
        # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        #         isinstance(self.task, tasks.names['bag-alone-open'])):
        #   removed_ids.append(self.task.zone_id)

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)

        info['cam_configs'] = tuple()
        for i, config in enumerate(self.ee_cams):
            info['cam_configs'] += (config,)
        for i, config in enumerate(self.agent_cams):
            info['cam_configs'] += (config,)

        return info

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            p.stepSimulation()
        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = defaultdict(tuple)
        for i, config in enumerate(self.ee_cams):
            color, depth, segm = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['segm'] += (segm,)

        for i, config in enumerate(self.agent_cams):

            color, depth, segm = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)
            obs['segm'] += (segm,)

        return obs

    def update_ee_cam(self):
        ee_pose = self.get_ee_pose()
        shift_vec = np.array([-0.08, -0.016, -0.12])
        n_ee_cams = len(self.ee_cams)
        for i in range(n_ee_cams):
            r1 = R.from_quat(ee_pose[1]).as_matrix()
            r2 = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
            r3 = R.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_matrix()
            # r3 = r2.apply(r1.as_rotvec())
            rt = r1 @ r2 @ r3
            rt = R.from_matrix(rt)
            rt_quat = rt.as_quat()
            self.ee_cams[i]['rotation'] = rt_quat
            new_pos = np.array(ee_pose[0])  # in w
            shift_vec_world = rt.as_matrix() @ shift_vec
            new_pos += shift_vec_world
            self.ee_cams[i]['position'] = tuple(new_pos)

    def get_ee_pose(self):
        return p.getLinkState(self.ur5, self.ee_tip)[0:2]


class ContinuousEnvironment(Environment):
    """A continuous environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Redefine action space, assuming it's a suction-based task. We'll override
        # it in `reset()` if that is not the case.
        self.position_bounds = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Dict({
            'move_cmd':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'suction_cmd': gym.spaces.Discrete(2),  # Binary 0-1.
            'acts_left': gym.spaces.Discrete(1000),
        })

    def set_task(self, task):
        super().set_task(task)

        # Redefine the action-space in case it is a pushing task. At this point, the
        # ee has been instantiated.
        if self.task.ee == Spatula:
            self.action_space = gym.spaces.Dict({
                'move_cmd':
                    gym.spaces.Tuple(
                        (self.position_bounds,
                         gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
                'slowdown_cmd': gym.spaces.Discrete(2),  # Binary 0-1.
                'acts_left': gym.spaces.Discrete(1000),
            })

    def step(self, action=None):
        if action is not None:
            timeout = self.task.primitive(self.movej, self.movep, self.ee, action)

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                self.update_ee_cam()
                obs = self._get_obs()
                info = copy.deepcopy(self.info)
                return obs, 0.0, True, info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            p.stepSimulation()

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        task_done = self.task.done()
        if action is not None:
            done = task_done and action['acts_left'] == 0
        else:
            done = task_done

        # Add ground truth robot state into info. and dynamic camera config
        # update before getting info
        self.update_ee_cam()
        info.update(self.info)
        obs = self._get_obs()

        return obs, reward, done, info


class RavensWrapper(gym.Wrapper):
    """A wrapper for the Ravens environment."""
    # add typing to init
    def __init__(self, env, task, stage: str = "train", image_size: List = (160, 120) , world_scale: float = 1.0 , coord_trans: List = None):

        self.env = env
        self.task = task
        self.task.mode = stage

        self.image_size = tuple(image_size)
        self.world_scale = world_scale

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self._coord_trans = torch.diag(
            torch.tensor(coord_trans if coord_trans is not None else [1, -1, -1, 1], dtype=torch.float32)
        )
        self.env.set_task(self.task)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.preprocess_obs(obs, self.env.info["cam_configs"])
        return obs

    @staticmethod
    def get_bbox(mask):
        """Get the bounding box of an object in the mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rnz = np.where(rows)[0]
        cnz = np.where(cols)[0]
        if len(rnz) == 0:
            raise RuntimeError(
                "ERROR: Bad image please investigate!"
            )
        rmin, rmax = rnz[[0, -1]]
        cmin, cmax = cnz[[0, -1]]
        bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
        return bbox

    def preprocess_obs(self, obs, configs=None):
        # Preprocess the observation.
        obs_preprocessed = defaultdict(list)
        # TODO(karim): add intrinsics per image

        for view in range(len(obs["color"])):
            #loop through all the views

            obs_preprocessed["images"].append(self.image_to_tensor(obs["color"][view]))
            mask = obs["segm"][view]
            mask_tensor = self.mask_to_tensor(mask)
            obs_preprocessed["masks"].append(mask_tensor)
            bbox = self.get_bbox(mask)
            obs_preprocessed["bbox"].append(bbox)
            if configs is not None:
                pose = pose_from_config(configs[view])
                #convert pose to tensor
                pose = torch.tensor(pose, dtype=torch.float32)
                pose = pose @ self._coord_trans if self._coord_trans is not None else pose
                obs_preprocessed["poses"].append(pose)

        # stack all elements in obs_preprocessed
        for key in obs_preprocessed.keys():
            obs_preprocessed[key] = torch.stack(obs_preprocessed[key])

        # add f, cx, cy from  configs
        if configs is not None:
            obs_preprocessed["focal"] = torch.tensor(configs[0]["intrinsics"][0], dtype=torch.float32)
            obs_preprocessed["c"] = torch.tensor([configs[0]["intrinsics"][2], configs[0]["intrinsics"][5]], dtype=torch.float32)

        if obs_preprocessed["images"].shape[-2:] != self.image_size:
            scale = self.image_size[0] / obs_preprocessed["images"].shape[-2]
            obs_preprocessed["focal"] *= scale
            obs_preprocessed["c"] *= scale
            obs_preprocessed["bbox"] *= scale

            obs_preprocessed["images"] = F.interpolate(obs_preprocessed["images"], size=self.image_size, mode="area")
            obs_preprocessed["masks"] = F.interpolate(obs_preprocessed["masks"], size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            obs_preprocessed["focal"] *= self.world_scale
            obs_preprocessed["poses"][:, :3, 3] *= self.world_scale

        obs_preprocessed = self.add_batch_dim(obs_preprocessed)
        return obs_preprocessed

    def add_batch_dim(self, obs):
        for key in obs.keys():
            obs[key] = obs[key].unsqueeze(0)
        return obs

    def process_action(self, action):
        action_processed = {}
        for key in self.env.action_space:
            # check if space is tuple or not
            if isinstance(self.env.action_space[key], gym.spaces.Tuple):
                # get len of action space if tuple
                len_space = len(self.env.action_space[key])
                # check if action is discrete get max value
                action_processed[key] = tuple([action[f"{key}_{str(i)}"].squeeze().detach().cpu().numpy() for i in range(len_space)])
            else:
                action_processed[key] = action[key].squeeze().detach().cpu().numpy()
        #convert actions from tensor to numpy
        return action_processed

    def step(self, action):
        #preprocess action
        action = self.process_action(action)
        #print(action, "action is after processing")
        obs, reward, done, info = self.env.step(action)
        # process obs
        obs = self.preprocess_obs(obs, info["cam_configs"])
        return obs, reward, done, info

    def get_info(self):
        return self.env.info

    def set_task(self, task=None):
        if task is not None:
            self.task = task
        self.env.set_task(self.task)

if __name__ == '__main__':
    # assets_root = 'ravens/ravens/environments/assets'
    # task_name = 'block_insertion'
    # mode = "train"
    # n = 3
    # steps_per_seg = 2
    # continuous = True
    # disp = True
    # shared_memory = False
    #
    # # Collect training data from oracle demonstrations.
    # env_cls = ContinuousEnvironment if continuous else Environment
    # env = env_cls(
    #     assets_root,
    #     disp=disp,
    #     shared_memory=shared_memory,
    #     hz=480)
    #
    # # Initialize scripted oracle agent and dataset.
    # task = tasks.names[task_name.replace("_", '-')](continuous=continuous)
    # task.mode = mode if mode != 'val' else 'test'
    # agent = task.oracle(env, steps_per_seg=steps_per_seg)
    #
    # # Train seeds are even and test seeds are odd.
    # cur_episode = 0
    # seed = -2
    # # Determine max steps per episode.
    # max_steps = task.max_steps
    # if continuous:
    #     max_steps *= (steps_per_seg * agent.num_poses)
    #
    # while cur_episode < n:
    #     print(f'Oracle demonstration: {cur_episode + 1}/{n}')
    #     episode, total_reward = [], 0
    #     seed += 2
    #     np.random.seed(seed)
    #     env.set_task(task)
    #     obs = env.reset()
    #     info = None
    #     reward = 0
    #     # print(f'maxstep is {max_steps}')
    #     for _ in range(max_steps):
    #         # print(f'info in start of episode: {info["d_poses"] if info is not None else None}')
    #         act = agent.act(obs, info)
    #         episode.append((obs, act, reward, copy.deepcopy(info)))
    #         obs, reward, done, info = env.step(act)
    #         # print(f'info after_step start of episode: {info["d_poses"]}')
    #         total_reward += reward
    #         print(f'Total Reward: {total_reward} Done: {done}')
    #         if done:
    #             break
    #         # print(obs['color'])
    #     episode.append((obs, act, reward, info))
    #     cur_episode += 1


    print("testing the wrapper environment")
    wrapper_cfg = OmegaConf.load("configs/ravens_wrapper.yaml")
    wrapped_env = hydra.utils.instantiate(wrapper_cfg.wrapper)
    n = 3
    steps_per_seg = 2
    agent = wrapped_env.task.oracle(wrapped_env, steps_per_seg=steps_per_seg)

    # Train seeds are even and test seeds are odd.
    cur_episode = 0
    seed = -2
    # Determine max steps per episode.
    max_steps = wrapped_env.task.max_steps
    max_steps *= (steps_per_seg * agent.num_poses)

    while cur_episode < n:
        print(f'Oracle demonstration: {cur_episode + 1}/{n}')
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        wrapped_env.env.set_task(wrapped_env.task)
        obs = wrapped_env.reset()
        info = None
        reward = 0
        # print(f'maxstep is {max_steps}')
        for _ in range(max_steps):
            # print(f'info in start of episode: {info["d_poses"] if info is not None else None}')
            act = agent.act(obs, info)
            print(f"action is {act}")
            act["move_cmd_0"] = act["move_cmd"][0]
            act["move_cmd_1"] = act["move_cmd"][1]
            episode.append((obs, act, reward, copy.deepcopy(info)))
            obs, reward, done, info = wrapped_env.step(act)
            # print(f'info after_step start of episode: {info["d_poses"]}')
            total_reward += reward
            print(f'Total Reward: {total_reward} Done: {done}')
            if done:
                break
            # print(obs['color'])
        episode.append((obs, act, reward, info))
        cur_episode += 1



