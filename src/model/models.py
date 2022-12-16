"""
Main model implementation
"""
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os
import numpy as np
import os.path as osp
import warnings


from pytorch3d.structures import Volumes, Pointclouds
from pytorch3d.ops import add_pointclouds_to_volumes

class PixelNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.use_encoder = conf.get("use_encoder", True)
        if self.use_encoder:
            self.encoder = make_encoder(conf["encoder"])
            self.encoder_type = conf.encoder.get("type", "spatial")

        self.use_xyz = conf.get("use_xyz", False)

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get("use_viewdirs", False)

        # Global image features?
        self.use_global_encoder = conf.get("use_global_encoder", False)

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = make_encoder(conf["global_encoder"]) #ImageEncoder.from_conf(conf["global_encoder"])
            self.global_encoder_type = conf.global_encoder.get("type", "field")
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        self.encoder_latent_size = self.encoder.latent_size if self.use_encoder else 0
        self.global_encoder_latent_size = self.global_encoder.latent_size if self.use_global_encoder else 0

        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, allow_empty=True
        )
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = self.mlp_coarse.d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=None, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        assert self.use_encoder or self.use_global_encoder
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            #if self.global_encoder_type == "field":
            self.pass_to_global_encoder(poses=self.poses,
                                        c=self.c,
                                        focal=self.focal,
                                        num_views_per_obj=self.num_views_per_obj,
                                        image_shape=self.image_shape,
                                        num_objs=self.num_objs)

        if self.use_encoder:
            self.encoder(images)
        if self.use_global_encoder:
            self.global_encoder(images)

    def pass_to_global_encoder(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.global_encoder, k, v)

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

    def get_pnts(self):
        #TODO(Karim) add start and end voxel size to the config

        dim_grid = []
        for s, e in zip(self.start, self.end):
            dim_grid.append(torch.linspace(start=s, end=e, steps=int((e - s) / self.voxel_size)))
        dim_grid = torch.meshgrid(dim_grid)
        dim_grid = torch.stack(dim_grid, dim=-1).flatten(end_dim=-2)

        return dim_grid.to(self.device)

    def construct_grid(self):

        xyz = self.get_pnts()
        xyz = xyz[None, ...].expand(self.num_objs, -1, -1)

        SB, B, _ = xyz.shape
        NS = self.num_views_per_obj
        xyz_rep = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
        # Transform query points into the camera spaces of the input views
        xyz_local = self.transform_to_local(xyz_rep)
        uv = self.transform_to_cam(xyz_local)
        latent = self.encoder.index(
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
        )#.to(self.device) #TODO could lead to error

        updated_volumes = add_pointclouds_to_volumes(
            pointclouds=pointclouds,
            initial_volumes=initial_volumes,
            mode="trilinear",
        )
        vol_feats = updated_volumes.features()
        vol_feats = vol_feats.permute(0, 1, 4, 3, 2)
        return vol_feats

    def get_latent(self):
        global_latent, latent = None, None
        if self.use_encoder:
            latent = self.encoder.latent
            if latent.size(0) != self.num_objs:
                latent = latent.view(self.num_objs, self.num_views_per_obj, *latent.shape[1:])
                latent = latent.mean(dim=1)
            if len(latent.size()) > 2:
                rem = tuple(range(2, len(latent.size())))
                latent = latent.mean(dim=rem)
        if self.use_global_encoder:
            global_latent = self.global_encoder.latent
            if global_latent.size(0) != self.num_objs:
                rem_sz = global_latent.shape[1:]
                global_latent = global_latent.view(self.num_objs, self.num_views_per_obj, *rem_sz)
                global_latent = global_latent.mean(dim=1)

        if self.use_global_encoder and self.use_encoder:
            out = torch.cat((latent, global_latent), dim=-1)
        else:
            out = latent if self.use_encoder else global_latent

        if self.stop_encoder_grad:
            out = out.detach()
        return out

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3)
            xyz = self.transform_to_local(xyz)
            xyz_rot = xyz - self.poses[:, None, :3, 3]

            if self.d_in > 0:
                # * Encode the xyz coordinates
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature
            #print('point stats in camera coord:', xyz.reshape(-1, 3).max(dim=0).values, xyz.reshape(-1, 3).min(dim=0).values, xyz.reshape(-1, 3).mean(dim=0).values,  '\n')
            if self.use_encoder:
                # Grab encoder's latent code.
                uv = self.transform_to_cam(xyz)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.encoder_latent_size
                )  # (SB * NS * B, latent)

                if self.d_in == 0:
                    # z_feature not needed
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                # Concat global latent code if enabled
                global_latent = self.global_encoder.latent

                if self.stop_encoder_grad:
                    global_latent = global_latent.detach()

                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            # Interpret the output
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)
        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
