import pybullet as p
import numpy as np

def pose_from_config(config):
    """Converts a config dict to a pose matrix."""
    rotation = p.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    position = np.float32(config['position'])
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotm #*np.array([1., -1., -1.]) to convert to openGL
    pose[:3, 3] = position
    return pose

