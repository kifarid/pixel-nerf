import numpy as np
import pytorch_lightening as pl
import torch
import util
from dotmap import DotMap
from model import make_model, loss
from render import NeRFRenderer


# TODO move to parser to main, set z near and far in trainer from dset and learning rate
# TODO move data set to main too
#
# def extra_args(parser):
#     parser.add_argument(
#         "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
#     )
#     parser.add_argument(
#         "--nviews",
#         "-V",
#         type=str,
#         default="1",
#         help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
#     )
#
#     parser.add_argument(
#         "--views_per_scene",
#         type=int,
#         default=3,
#         help="Number of views to consider per 1 scene",
#     )
#     parser.add_argument(
#         "--freeze_enc",
#         action="store_true",
#         default=None,
#         help="Freeze encoder weights and only train MLP",
#     )
#
#     parser.add_argument(
#         "--no_bbox_step",
#         type=int,
#         default=100000,
#         help="Step to stop using bbox sampling",
#     )
#     parser.add_argument(
#         "--fixed_test",
#         action="store_true",
#         default=None,
#         help="Freeze encoder weights and only train MLP",
#     )
#     return parser
#
#
# args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
# dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, views_per_scene=args.views_per_scene)
# print(
#     "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
# )
#

class NeRFAE(pl.LightningModule):
    def __init__(self,
                 conf,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.net = make_model(conf["model"])  # .to(device=self.device)
        self.net.stop_encoder_grad = conf["model"].freeze_enc
        if conf["model"].freeze_enc:
            print("Encoder frozen")
            self.net.encoder.eval()

        self.lambda_coarse = conf["loss"].get("lambda_coarse")
        self.lambda_fine = conf["loss"].get("lambda_fine", 1.0)
        self.no_bbox_step = conf.conf["renderer"].no_bbox_step
        self.use_bbox = self.no_bbox_step > 0
        # self.learning_rate = conf["model"].learning

        self.renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=False)  # .to(
        #     device=device
        # )

        # Parallize
        self.render_par = self.renderer.bind_parallel(self.net, conf["renderer"].gpus).eval()

        self.nviews = conf["model"].nviews
        self.nviews = list(map(int, self.nviews.split()))
        self.loss_from_config(conf["loss"])
        self.gamma = conf["model"].gamma

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
        all_images = data["images"]  # .to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"]  # .to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
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
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
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
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds]  # .to(
            #    device=device
            # )  # (ray_batch_size, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, ray_batch_size, 3)
        all_rays = torch.stack(all_rays)  # (SB, ray_batch_size, 8)

        image_ord = image_ord  # .to(device)
        src_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = all_poses = all_images = None

        self.net.encode(
            src_images,
            src_poses,
            all_focals,  # .to(device=device),
            c=all_c if all_c is not None else None,  # all_c.to(device=device)
        )

        render_dict = DotMap(self.render_par(all_rays, want_weights=True, ))
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine

        loss = rgb_loss
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, batch_idx):
        return self.calc_losses(data, is_train=True)

    def eval_step(self, data, batch_idx):
        self.renderer.eval()
        losses = self.calc_losses(data, is_train=False)
        self.renderer.train()
        return losses

    def vis_step(self, data, batch_idx, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx]  # .to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx]  # .to(device=device)  # (NV, 4, 4)
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
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        self.renderer.train()
        return vis, vals

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


if __name__ == "__main__":
    dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, views_per_scene=args.views_per_scene)
    print(
        "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
    )