import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.RealWorldDataset import TeleopData
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    @torch.no_grad()
    def forward(self, x):
        n, c, h, w = x.size()
        #assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        arange_h = torch.linspace(-1.0 + eps_h,
                                1.0 - eps_h,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange_w = torch.linspace(-1.0 + eps_w,
                                1.0 - eps_w,
                                w + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:w]

        arange_h = arange_h.unsqueeze(0).repeat(1, w).unsqueeze(2)
        arange_w = arange_w.unsqueeze(1).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange_h, arange_w], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomShiftsAugNew(nn.Module):
    def __init__(self, pad=2, consistent=False):
        super().__init__()
        self.pad = pad
        self.consistent = consistent

    @torch.no_grad()
    def forward(self, x):
        orig_size = x.size()
        if len(orig_size) > 4:
            x = x.permute(1, 0, 2, 3, 4)
            orig_size = x.size()
            t, n, c, h, w = x.size()
        else:
            n, c, h, w = x.size()
        #(n, c, h, w)
        #assert h == w

        padding = tuple([self.pad] * 4 + [0, 0])  # len(x.size()))
        x = F.pad(x, padding, 'replicate')
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        arange_h = torch.linspace(-1.0 + eps_h,
                                  1.0 - eps_h,
                                  h + 2 * self.pad,
                                  device=x.device,
                                  dtype=x.dtype)[:h]
        arange_w = torch.linspace(-1.0 + eps_w,
                                  1.0 - eps_w,
                                  w + 2 * self.pad,
                                  device=x.device,
                                  dtype=x.dtype)[:w]
        #print(arange_h.shape, arange_w.shape)
        arange_h = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        #print(arange_h.shape, arange_w.shape)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        #base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        base_grid = base_grid[None, None, ...].repeat(t, n, 1, 1, 1) if len(x.size()) > 4 else base_grid.unsqueeze(
                0).repeat(n, 1, 1, 1)

        if self.consistent:
            shift = torch.randint(0,
                                  2 * self.pad + 1,
                                  size=(n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)
            shift = shift.repeat(t, 1, 1, 1, 1) if len(orig_size) > 4 else shift
        else:
            shift = torch.randint(0,
                                  2 * self.pad + 1,
                                  size=(t, n, 1, 1, 2) if len(orig_size) > 4 else (n, 1, 1, 2),
                                  device=x.device,
                                  dtype=x.dtype)

        shift *= 1.0 / (h + 2 * self.pad) + 1.0 / (w + 2 * self.pad)

        grid = base_grid + shift
        if len(orig_size) > 4:
            x = x.view((-1, *x.size()[2:]))
            grid = grid.view((-1, *grid.size()[2:]))

        samples = F.grid_sample(x,
                                grid,
                                padding_mode='zeros',
                                align_corners=False)
        # samples = samples.view(orig_size)
        if len(orig_size) > 4:
            samples = samples.view(orig_size)
            samples = samples.permute(1, 0, 2, 3, 4)
        else:
            return samples

        return samples


if __name__ == "__main__":
    random_image_batch = torch.rand(8, 3, 2, 3, 150, 200)
    random_shifts_aug = RandomShiftsAugNew(1)
    random_shifts_aug(random_image_batch.view(8*3*2, 3, 150, 200))
    # random_shifts_aug.consistent = True
    # random_shifts_aug(random_image_batch.view(8, 3*2, 3, 150, 200))
    frame_stack = 10
    dataset = TeleopData(
        "/Users/kfarid/Desktop/Education/MSc_Freiburg/research/robot_io/expert_data/val",
        frame_stack=5,
        frame_rate=2,
        views=[1, 2],
        relative=True,
        # image_size=(128, 128),
        # views_per_scene=3,
        # world_scale=0.1,

    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    for i, sample in enumerate(dataloader):

        images_aug_ = sample["images"].view(8*2*5, 3, 150, 200)
        images_aug_ = random_shifts_aug(images_aug_)
        # show images
        images_aug_ = images_aug_.view(8, 2, 5, 3, 150, 200)
        # convert to numpy
        print(images_aug_.shape)
        images_aug_ = images_aug_.permute(0, 1, 2, 4, 5, 3).cpu().numpy()
        images_view_1 = images_aug_[0, :, :, :, :, :]
        # add over height
        print(images_view_1.shape)
        iv = np.concatenate(images_view_1, axis=2)
        print(iv.shape)
        iv = np.concatenate(iv, axis=0)
        print(iv.shape)
        images_view_1 = iv
        plt.imshow(images_view_1)
        plt.show()




