import numpy as np
import pytorch_lightning as pl
import torch
import util
from dotmap import DotMap
from model import make_model, loss
from render import NeRFRenderer


class NeRFAE(pl.LightningModule):
    def __init__(self,
                 renderer_config,
                 model_config,
                 loss_config,
                 base_learning_rate=1e-4,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 logging=True,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.logging = logging
        self.net_type = model_config.get("type", "pixelnerf")
        self.net = make_model(model_config)  # .to(device=self.device)
        self.net.stop_encoder_grad = model_config.freeze_enc
        if model_config.freeze_enc:
            print("Encoder frozen")
            if model_config.use_encoder:
                self.net.encoder.eval()
                # freeze encoder
                for param in self.net.encoder.parameters():
                    param.requires_grad = False

            if model_config.use_global_encoder:
                self.net.global_encoder.eval()
                # freeze encoder
                for param in self.net.global_encoder.parameters():
                    param.requires_grad = False

        self.lambda_coarse = loss_config.get("lambda_coarse")
        self.lambda_fine = loss_config.get("lambda_fine", 1.0)
        self.lambda_vic = loss_config.get("lambda_vic", 0.0001)

        self.no_bbox_step = renderer_config.no_bbox_step
        self.use_bbox = self.no_bbox_step > 0
        # self.learning_rate = conf["model"].learning

        if self.net_type == "dnerf":
            self.renderer = DNeRFRenderer.from_conf(renderer_config, lindisp=False)
            self.lambda_flow = loss_config.get("lambda_flow", 0.0001)
        else:
            self.renderer = NeRFRenderer.from_conf(renderer_config, lindisp=False)

        # Parallize
        self.render_par = self.renderer.bind_parallel(self.net, renderer_config.gpus).eval()

        self.nviews = model_config.nviews
        self.nviews = list(map(int, self.nviews.split()))
        self.loss_from_config(loss_config)
        self.gamma = model_config.gamma

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.ray_batch_size = renderer_config.ray_batch_size
        self.z_near = renderer_config.z_near
        self.z_far = renderer_config.z_far

    def loss_from_config(self, lossconfig):

        self.lambda_coarse = lossconfig.get("lambda_coarse")
        self.lambda_fine = lossconfig.get("lambda_fine", 1.0)
        print(
            "lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(lossconfig["rgb"], True)
        fine_loss_conf = lossconfig["rgb"]
        if "rgb_fine" in lossconfig:
            print("using fine loss")
            fine_loss_conf = lossconfig["rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

    def calc_losses(self, data, is_train=True):
        if "images" not in data:
            return {}
        all_images = data["images"]  # .to(device=device)  # (SB, NV, 3, H, W) or  (SB, T, NV, 3, H, W)

        if len(all_images.shape) > 5:
            all_images = data["images"][:, -1, ...]

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"][:, -1, ...]  if len(data["poses"].shape) > 4 else data["poses"] #.to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox", None)  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and self.global_step >= self.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", self.global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1)).to(self.device)
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long).to(self.device)

        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)
            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, self.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (self.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
             device=self.device
            )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord.to(self.device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)
        src_focals = util.batched_index_select_nd(all_focals, image_ord) if len(all_focals.shape)>2 else all_focals

        if all_c is not None:
            src_cs = util.batched_index_select_nd(all_c, image_ord) if len(all_c.shape)>2 else all_c
        else:
            src_cs = None

        all_bboxes = all_poses = all_images = None
        self.net.encode(
            src_images,
            src_poses,
            src_focals,
            c=src_cs,
        )

        render_dict = DotMap(self.render_par(all_rays, want_weights=True, ))
        coarse = render_dict.coarse
        using_vic = self.render_par.renderer.vic_radius > 0
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        if using_vic:
            vic_loss = self.lambda_coarse*self.lambda_vic*(coarse.depth_vic_diff) #coarse.rgb_vic_mse
            if using_fine:
                vic_loss += self.lambda_fine*self.lambda_vic/2*(fine.depth_vic_diff) #fine.rgb_vic_mse
            loss_dict["rd_vic"] = vic_loss.item()
            rgb_loss+=vic_loss
        if self.net_type == "dnerf":
            flow_loss = self.lambda_coarse*self.lambda_flow*coarse.flow_dist
            if using_fine:
                flow_loss += self.lambda_fine*self.lambda_flow*fine.flow_dist
            loss_dict["flow"] = flow_loss.item()
            rgb_loss+=flow_loss
        
        loss = rgb_loss
        
        loss_dict["loss"] = loss
        return loss_dict

    def training_step(self, data, batch_idx):
        loss_dict = self.calc_losses(data, is_train=True)
        if self.logging:
            self.log("nerf_train_loss", loss_dict, on_step=True, on_epoch=True)
        return loss_dict

    def validation_step(self, data, batch_idx):
        self.renderer.eval()
        self.net.eval()
        losses = self.calc_losses(data, is_train=False)
        if self.logging:
            self.log("nerf_val_loss", losses, on_step=False, on_epoch=True)
        self.renderer.train()
        self.net.train()
        return losses

    def test_step(self, data, batch_idx):
        self.renderer.eval()
        self.net.eval()
        losses = self.calc_losses(data, is_train=False)
        if self.logging:
            self.log("nerf_test_loss", losses, on_step=False, on_epoch=True)
        self.renderer.train()
        self.net.train()
        return losses

    def log_images(self, data, idx=None, **kwargs):
        #print("in log images")
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx

        images = data["images"][batch_idx, -1] if len(data["images"].shape) >5 else data["images"][batch_idx, -1] # .to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx, -1] if len(data["poses"].shape) > 4 else data["poses"][batch_idx]  # .to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx: batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx: batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = self.nviews[torch.randint(0, len(self.nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        self.renderer.eval()
        source_views = (
            images_0to1[views_src]
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .reshape(-1, H, W, 3)
        )

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            focal = focal[:, views_src] if len(focal.shape) > 2 else focal
            if c is not None:
                c = c[:, views_src] if len(c.shape) > 2 else c

            self.net.encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal,  # .to(device=device),
                c=c if c is not None else None,  # .to(device=device)
            )

            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(self.render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        self.log("psnr", psnr, on_step=False, on_epoch=True)
        # set the renderer network back to train mode
        self.renderer.train()

        vis_t=torch.from_numpy(vis).unsqueeze(0).permute(0, 3, 1, 2)
        vis_dict = {"images": vis_t}
        return vis_dict

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def get_latent(self):
        return self.net.get_latent()

    def encode(self, images, poses, focal, c=None):
        return self.net.encode(images, poses, focal, c=c)

    def configure_optimizers(self):

        lr = self.learning_rate
        # Currently only Adam supported
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)  # self

        if self.gamma != 1.0:
            print("Setting up LambdaLR scheduler...")
            lr_scheduler = [
                {
                    'scheduler': torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optim, gamma=self.gamma),  # LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]

            return optim, lr_scheduler

        return optim

