import glob
import os
import pickle

import imageio
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class RDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
            self, path, stage="train", z_near=0.01, z_far=1.6, image_size=(120, 160), world_scale=1.0, views_per_scene=12
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        :param number of views to consider per scene
        """
        super().__init__()
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)
        self.views_per_scene = views_per_scene
        print(f'considering only the first {self.views_per_scene}')

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = z_near #0.1
        self.z_far = z_far #1.6
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))[:self.views_per_scene]
        mask_paths = sorted(glob.glob(os.path.join(dir_path, "segm", "*")))[:self.views_per_scene]
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))[:self.views_per_scene]

        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, mask_path, pose_path in zip(rgb_paths, mask_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                with tf.io.gfile.GFile(mask_path, 'rb') as f:
                    mask = pickle.load(f)
                    
                    if len(mask.shape) == 2:
                        mask = mask[..., None]
                    mask = mask[..., :1]
            else:
                mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            
            mask = mask.astype(np.float32)
            mask_tensor = self.mask_to_tensor(mask)
            #print(mask_tensor.max(), mask_tensor.min(), 'mask tensor data in dataset ravens', rgb_path, mask_path)
            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1) #width
            cols = np.any(mask, axis=0) #height 
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            #print(bbox, rgb_path)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)
        #print(self.image_size, all_imgs.shape, 'image desired and actual shapes')
        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")
        #print(all_bboxes.max(), all_bboxes.min(), 'min and max of bboxes after')
        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
