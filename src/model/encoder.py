"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import util
from model.custom_encoder import ConvEncoder
from model.vol_encoder import VolEncoder
import torch.autograd.profiler as profiler
import numpy as np
from util import repeat_interleave
#from model.model_util import make_mlp

from pytorch3d.structures import Volumes, Pointclouds
from pytorch3d.ops import add_pointclouds_to_volumes
import torch


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get("backbone"),
            pretrained=conf.get("pretrained", True),
            num_layers=conf.get("num_layers", 4),
            index_interp=conf.get("index_interp", "bilinear"),
            index_padding=conf.get("index_padding", "border"),
            upsample_interp=conf.get("upsample_interp", "bilinear"),
            feature_scale=conf.get("feature_scale", 1.0),
            use_first_pool=conf.get("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128, encode_cam=True):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)


        self.poses = None
        self.c = None
        self.focal = None
        self.num_views_per_object = None
        self.num_objs = None
        self.encode_cam = encode_cam

        # create mlp to process the output and camera parameters (pose, focal length, c)
        if self.encode_cam:

            self.g_mlp = nn.Sequential(
                nn.Linear(12+2+2 + latent_size, 128),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(128, 128), nn.ReLU()) for _ in range(1)],
                nn.Linear(128, latent_size)
            )

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)

        # process the output and camera parameters (pose, focal length, c) using g_mlp
        if self.poses is not None and self.encode_cam:
            # poses (SB*NS, 12)
            # c (SB, 2)
            # focal (SB, 2)
            f = repeat_interleave(self.focal,
                                  self.num_views_per_obj)
            c = repeat_interleave(self.c, self.num_views_per_obj)
            x = self.g_mlp(torch.cat([x, self.poses.flatten(1, -1), f, c], dim=-1))
            self.latent = x

        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get("backbone"),
            pretrained=conf.get("pretrained", True),
            latent_size=conf.get("latent_size", 128),
            encode_cam=conf.get("encode_cam", True),
        )


class FieldEncoder(nn.Module):
    """
    3D (Spatial/Pixel-aligned/global) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        start=np.array([0., 0., 0.]),
        end=np.array([1., 1., 1.]),
        voxel_size=0.01, 
        pnts_per_voxel=5

    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.start = start
        self.end = end
        self.voxel_size = voxel_size
        self.pnts_per_voxel = pnts_per_voxel
        self.grid = (self.end-self.start)//self.voxel_size
        self.grid = np.ceil(self.grid / 2) * 2
        grid_max = int(self.grid.max())
        self.vol_encoder = VolEncoder((self.latent_size, grid_max, grid_max, grid_max), d_latent = self.latent_size)
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)
        self.poses = None
        self.c = None
        self.focal = None
        self.num_views_per_object = None
        self.num_objs = None

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.unaligned_latent.shape[0] > 1:
                uv = uv.expand(self.unaligned_latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.unaligned_latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        self.device = x.device
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
       #x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.unaligned_latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.unaligned_latent = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )

            self.unaligned_latent = torch.cat(latents, dim=1)

        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0

        vol_feat_x = self.construct_grid()
        z_vec = self.vol_encoder(vol_feat_x)
        self.latent = z_vec

        return self.latent

    def transform_to_cam(self, xyz_local):
        NS = self.num_views_per_obj
        divisor = xyz_local[:, :, 2:]
        divisor = torch.where(torch.abs(divisor) > 1e-4, divisor, torch.full(divisor.size(), 1e-4).to(divisor))
        uv = -xyz_local[:, :, :2] / divisor  # (SB, B, 2)

        uv *= repeat_interleave(
            self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1
        )
        uv += repeat_interleave(
            self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1
        )  # (SB*NS, B, 2)
        return uv

    def transform_to_local(self, xyz):
        xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
            ..., 0
        ]
        xyz = xyz_rot + self.poses[:, None, :3, 3]
        return xyz

    def construct_grid(self):

        xyz = self.get_pnts()
        xyz = xyz[None, ...].expand(self.num_objs, -1, -1)

        SB, B, _ = xyz.shape
        NS = self.num_views_per_obj
        xyz_rep = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
        # Transform query points into the camera spaces of the input views
        xyz_local = self.transform_to_local(xyz_rep)
        uv = self.transform_to_cam(xyz_local)
        latent = self.index(
            uv, None, self.image_shape
        )  # (SB * NS, latent, B)
        # get another view of latent with num of objects and num of views per object
        latent = latent.view(SB, NS, -1, B).mean(dim=1)  # (SB, latent, B)

        pointclouds = Pointclouds(
                points=xyz, features=latent.permute(0, 2, 1).float())
        grid_max = int(self.grid.max())

        initial_volumes = Volumes(
            features=torch.zeros(latent.size(0), self.latent_size, *(grid_max, grid_max, grid_max)).float(),
            densities=torch.zeros(latent.size(0), 1, *(grid_max, grid_max, grid_max)).float(),
            volume_translation=-torch.from_numpy((self.end + self.start) / 2).float(),
            voxel_size=self.voxel_size,
        ).to(self.device)

        updated_volumes = add_pointclouds_to_volumes(
            pointclouds=pointclouds,
            initial_volumes=initial_volumes,
            mode="trilinear", #"nearest"
        )
        vol_feats = updated_volumes.features()
        vol_feats = vol_feats.permute(0, 1, 4, 3, 2)
        return vol_feats

    def get_pnts(self):
        dim_grid = []
        for s, e in zip(self.start, self.end):
            dim_grid.append(torch.linspace(start=s, end=e, steps=int(self.pnts_per_voxel*(e - s) / self.voxel_size)))
        dim_grid = torch.meshgrid(dim_grid)
        dim_grid = torch.stack(dim_grid, dim=-1).flatten(end_dim=-2)

        return dim_grid.to(self.device)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get("backbone"),
            pretrained=conf.get("pretrained", True),
            num_layers=conf.get("num_layers", 4),
            voxel_size= conf.get("voxel_size", 0.03125),
            pnts_per_voxel = conf.get("pnts_per_voxel", 2),
            start=np.array(conf.get("start", [0., 0., 0.])),
            end=np.array(conf.get("end", [1., 1., 1.])),
            index_interp=conf.get("index_interp", "bilinear"),
            index_padding=conf.get("index_padding", "border"),
            upsample_interp=conf.get("upsample_interp", "bilinear"),
            feature_scale=conf.get("feature_scale", 1.0),
            use_first_pool=conf.get("use_first_pool", True),
        )

