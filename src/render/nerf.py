"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import numpy as np
import torch.nn.functional as F
import util
import torch.autograd.profiler as profiler
from torch.nn import DataParallel
from dotmap import DotMap
import datetime


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(self, rays, want_weights=False):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net, rays, want_weights=want_weights and not self.simple_output
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        vic_radius=0,
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std
        self.vic_radius = vic_radius

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def composite(self, model, rays, z_samp, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape
            print("B, K:", B, K)
            # print('z samp in nerf renderer:', z_samp.min(dim=0).values, z_samp.min(dim=0).values, '\n')

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            #  if far:
            #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            # (B, K, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            if self.vic_radius > 0:
                # Generate random noise from a normal distribution with mean 0 and standard deviation 0.1
                pnts_sz = points.size()
                noise = torch.randn(*pnts_sz) * self.vic_radius
                points_vic = points + noise

            #print("initial points shape:", points.shape)
            if torch.isnan(points).any() or torch.isinf(points).any():
                print('problem is only in original points have nan values before model')
            # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            # points_stats = points.detach().cpu().numpy()
            # with open(f'xyz_{now}.npy', 'wb') as f:
            #    np.save(f, points_stats)
            # print(f'max-min in z is {np.amax(points_stats[...,2])}, {np.amin(points_stats[...,2])} \n')
            # print(f'max-min in x is {np.amax(points_stats[...,0])}, {np.amin(points_stats[...,0])} \n')
            # print(f'max-min in y is {np.amax(points_stats[...,1])}, {np.amin(points_stats[...,1])} \n')

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs
            val_all = []

            if self.vic_radius > 0:
                val_all_vic = []

            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                if self.vic_radius > 0:
                    points_vic = points_vic.reshape(
                        sb, -1, 3
                    )
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0
            print("points shape:", points.shape)
            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if self.vic_radius > 0:
                split_points_vic = torch.split(points_vic, eval_batch_size, dim=eval_batch_dim)
            #print([(i, p.shape) for i, p in enumerate(split_points)])
            #print("split points shape:", eval_batch_size, eval_batch_dim, points.shape)
            if use_viewdirs:
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )
                for pnts, dirs in zip(split_points, split_viewdirs):
                    model_o = model(pnts, coarse=coarse, viewdirs=dirs)
                    val_all.append(model_o)
                if self.vic_radius > 0:
                    for pnts_vic, dirs in zip(split_points_vic, split_viewdirs):
                        model_o = model(pnts_vic, coarse=coarse, viewdirs=dirs)
                        val_all_vic.append(model_o)
            else:
                for pnts in split_points:
                    model_o = model(pnts, coarse=coarse)
                    val_all.append(model_o)

                if self.vic_radius > 0:
                    for pnts_vic in split_points_vic:
                        model_o = model(pnts_vic, coarse=coarse)
                        val_all_vic.append(model_o)

            points = None
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            print(f"out before shape: {out.shape}, and val all  {val_all[0].shape}!!!!!!!!!!!!!")
            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)
            print(f"out shape: {out.shape}!!!!!!!!!!!!!")
            print(f" B: {B}, K: {K}, sb: {sb}")

            rgb_final, depth_final, weights = self.post_process_output(out, deltas, z_samp)

            if self.vic_radius > 0:
                out_vic = torch.cat(val_all_vic, dim=eval_batch_dim)
                out_vic = out_vic.reshape(B, K, -1)
                rgb_mse, alpha_mse = self.get_diff(out, out_vic, deltas)
                return (weights, rgb_final, depth_final), (rgb_mse, alpha_mse)


            return (
                weights,
                rgb_final,
                depth_final,
            )

    def get_diff(self, out, out_vic, deltas):
        """return the difference in the rgb and depth values between the two outputs"""
        rgbs = out[..., :3]
        rgbs_vic = out_vic[..., :3]
        rgb_mse = torch.mean(torch.pow(rgbs - rgbs_vic, 2))
        sigmas = out[..., 3]
        sigmas_vic = out_vic[..., 3]
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
        alphas_vic = 1 - torch.exp(-deltas * torch.relu(sigmas_vic))
        alpha_mse = torch.mean(torch.pow(alphas - alphas_vic, 2))
        return rgb_mse, alpha_mse

    def post_process_output(self, out, deltas, z_samp):
        """
        Args:
            out: (B, K, 4 or 5)
            deltas: (B, K)
            z_samp: (B, K)
        Returns:
            rgb_final: (B, 3)
        """
        rgbs = out[..., :3]  # (B, K, 3)
        sigmas = out[..., 3]  # (B, K)
        if self.training and self.noise_std > 0.0:
            sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
        deltas = None
        sigmas = None
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)  # (B)
        weights = alphas * T[:, :-1]  # (B, K)
        alphas = None
        alphas_shifted = None

        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
        depth_final = torch.sum(weights * z_samp, -1)  # (B)
        if self.white_bkgd:
            # White background
            pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

        return rgb_final, depth_final, weights

    def forward(
        self, model, rays, want_weights=False,
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            rays = rays.reshape(-1, 8)  # (SB * B, 8)
            z_coarse = self.sample_coarse(rays)  # (B, Kc)
            coarse_composite = self.composite(
                model, rays, z_coarse, coarse=True, sb=superbatch_size,
            )
            if self.vic_radius > 0:
                coarse_composite, coarse_composite_vic = coarse_composite

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights,
                ),
            )

            if self.vic_radius > 0:
                outputs.coarse.rgb_vic_mse = coarse_composite_vic[0]
                outputs.coarse.depth_vic_mse = coarse_composite_vic[1]

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                fine_composite = self.composite(
                    model, rays, z_combine_sorted, coarse=False, sb=superbatch_size,
                )

                if self.vic_radius > 0:
                    fine_composite, fine_composite_vic = fine_composite


                outputs.fine = self._format_outputs(
                    fine_composite, superbatch_size, want_weights=want_weights,
                )

                if self.vic_radius > 0:
                    outputs.fine.rgb_vic_mse = fine_composite_vic[0]
                    outputs.fine.depth_vic_mse = fine_composite_vic[1]

            #print("finished rendering")
            return outputs

    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False,
    ):
        weights, rgb, depth = rendered_outputs
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
        ret_dict = DotMap(rgb=rgb, depth=depth)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict


    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False, eval_batch_size=100000):
        return cls(
            conf.get("n_coarse", 128),
            conf.get("n_fine", 0),
            n_fine_depth=conf.get("n_fine_depth", 0),
            noise_std=conf.get("noise_std", 0.0),
            depth_std=conf.get("depth_std", 0.01),
            white_bkgd=conf.get("white_bkgd", white_bkgd),
            lindisp=lindisp,
            eval_batch_size=conf.get("eval_batch_size", eval_batch_size),
            vic_radius=conf.get("vic_radius", 0.001),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the 
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped


# class DNeRFRenderer(NeRFRenderer):
#     """
#     NeRF differentiable renderer
#     :param n_coarse number of coarse (binned uniform) samples
#     :param n_fine number of fine (importance) samples
#     :param n_fine_depth number of expected depth samples
#     :param noise_std noise to add to sigma. We do not use it
#     :param depth_std noise for depth samples
#     :param eval_batch_size ray batch size for evaluation
#     :param white_bkgd if true, background color is white; else black
#     :param lindisp if to use samples linear in disparity instead of distance
#     """
#
#     def __init__(
#         self,
#         n_coarse=128,
#         n_fine=0,
#         n_fine_depth=0,
#         noise_std=0.0,
#         depth_std=0.01,
#         eval_batch_size=100000,
#         white_bkgd=False,
#         lindisp=False,
#         vic_radius=0,
#     ):
#
#         super().__init__()
#         self.n_coarse = n_coarse
#         self.n_fine = n_fine
#         self.n_fine_depth = n_fine_depth
#
#         self.noise_std = noise_std
#         self.depth_std = depth_std
#         self.vic_radius = vic_radius
#
#         self.eval_batch_size = eval_batch_size
#         self.white_bkgd = white_bkgd
#         self.lindisp = lindisp
#         if lindisp:
#             print("Using linear displacement rays")
#         self.using_fine = n_fine > 0
#         self.register_buffer(
#             "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
#         )
#
#     def composite(self, model, rays, z_samp, coarse=True, sb=0):
#         """
#         Render RGB and depth for each ray using NeRF alpha-compositing formula,
#         given sampled positions along each ray (see sample_*)
#         :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
#         should also support 'coarse' boolean argument
#         :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
#         :param z_samp z positions sampled for each ray (B, K)
#         :param coarse whether to evaluate using coarse NeRF
#         :param sb super-batch dimension; 0 = disable
#         :return weights (B, K), rgb (B, 3), depth (B)
#         """
#         with profiler.record_function("renderer_composite"):
#             B, K = z_samp.shape
#             # print('z samp in nerf renderer:', z_samp.min(dim=0).values, z_samp.min(dim=0).values, '\n')
#
#             deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
#             #  if far:
#             #      delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
#             delta_inf = rays[:, -1:] - z_samp[:, -1:]
#             deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)
#
#             # (B, K, 3)
#             points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
#             points = points.reshape(-1, 3)  # (B*K, 3)
#
#             if self.vic_radius > 0:
#                 # Generate random noise from a normal distribution with mean 0 and standard deviation 0.1
#                 pnts_sz = points.size()
#                 noise = torch.randn(*pnts_sz) * self.vic_radius
#                 points_vic = points + noise
#
#             #print("initial points shape:", points.shape)
#             if torch.isnan(points).any() or torch.isinf(points).any():
#                 print('problem is only in original points have nan values before model')
#
#             use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs
#             val_all = []
#             val_all_flow = []
#
#             if self.vic_radius > 0:
#                 val_all_vic = []
#                 val_all_flow_vic = []
#
#
#             if sb > 0:
#                 points = points.reshape(
#                     sb, -1, 3
#                 )  # (SB, B'*K, 3) B' is real ray batch size
#                 if self.vic_radius > 0:
#                     points_vic = points_vic.reshape(
#                         sb, -1, 3
#                     )
#                 eval_batch_size = (self.eval_batch_size - 1) // sb + 1
#                 eval_batch_dim = 1
#             else:
#                 eval_batch_size = self.eval_batch_size
#                 eval_batch_dim = 0
#
#             split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
#             if self.vic_radius > 0:
#                 split_points_vic = torch.split(points_vic, eval_batch_size, dim=eval_batch_dim)
#
#             if use_viewdirs:
#                 dim1 = K
#                 viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
#                 if sb > 0:
#                     viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
#                 else:
#                     viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
#                 split_viewdirs = torch.split(
#                     viewdirs, eval_batch_size, dim=eval_batch_dim
#                 )
#
#                 for pnts, dirs in zip(split_points, split_viewdirs):
#
#                     model_o = model(pnts, coarse=coarse, viewdirs=dirs)
#                     model_o_flow = model_o[1]
#                     model_o = model_o[0]
#                     val_all_flow.append(model_o_flow)
#                     val_all.append(model_o)
#
#                 if self.vic_radius > 0:
#                     for pnts_vic, dirs in zip(split_points_vic, split_viewdirs):
#                         model_o = model(pnts_vic, coarse=coarse, viewdirs=dirs)
#
#                         model_o_flow = model_o[1]
#                         model_o = model_o[0]
#                         val_all_flow_vic.append(model_o_flow)
#                         val_all_vic.append(model_o)
#             else:
#                 for pnts in split_points:
#                     model_o = model(pnts, coarse=coarse)
#                     model_o_flow = model_o[1]
#                     model_o = model_o[0]
#                     val_all_flow.append(model_o_flow)
#                     val_all.append(model_o)
#
#                 if self.vic_radius > 0:
#                     for pnts in split_points_vic:
#                         model_o = model(pnts, coarse=coarse)
#
#                         model_o_flow = model_o[1]
#                         model_o = model_o[0]
#                         val_all_flow_vic.append(model_o_flow)
#                         val_all_vic.append(model_o)
#
#             points = None
#             viewdirs = None
#             # (B*K, 4) OR (SB, B'*K, 4)
#             out = torch.cat(val_all, dim=eval_batch_dim)
#             print(out.shape,val_all[0].shape, val_all[-1].shape, eval_batch_dim)
#             out = out.reshape(B, K, -1)  # (B, K, 4 or 5)
#             print(val_all_flow[0].shape, val_all_flow[-1].shape, eval_batch_dim)
#             out_flow = torch.cat(val_all_flow, dim=eval_batch_dim)
#             out_flow = out_flow.reshape(B, K, -1)  # (B, K, 3)
#             rgb_final, depth_final, weights = self.post_process_output(out, deltas, z_samp)
#
#             if self.vic_radius > 0:
#                 out_vic = torch.cat(val_all_vic, dim=eval_batch_dim)
#                 out_vic = out_vic.reshape(B, K, -1)
#                 out_flow_vic = torch.cat(val_all_flow_vic, dim=eval_batch_dim)
#                 out_flow_vic = out_flow_vic.reshape(B, K, -1) # (B, K, 3)
#                 rgb_mse, depth_mse = self.get_diff(out, out_vic)
#                 flow_dist = self.get_flow_diff(out_flow, out_flow_vic)
#                 #return (weights, rgb_final, depth_final), (rgb_mse, depth_mse)
#                 return (weights, rgb_final, depth_final, out_flow), (rgb_mse, depth_mse, flow_dist)
#
#
#             return (
#                 weights,
#                 rgb_final,
#                 depth_final,
#                 out_flow,
#             )
#
#     def forward(
#         self, model, rays, want_weights=False,
#     ):
#         """
#         :model nerf model, should return (SB, B, (r, g, b, sigma))
#         when called with (SB, B, (x, y, z)), for multi-object:
#         SB = 'super-batch' = size of object batch,
#         B  = size of per-object ray batch.
#         Should also support 'coarse' boolean argument for coarse NeRF.
#         :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
#         :param want_weights if true, returns compositing weights (SB, B, K)
#         :return render dict
#         """
#         with profiler.record_function("renderer_forward"):
#
#             assert len(rays.shape) == 3
#             superbatch_size = rays.shape[0]
#             rays = rays.reshape(-1, 8)  # (SB * B, 8)
#             z_coarse = self.sample_coarse(rays)  # (B, Kc)
#             coarse_composite = self.composite(
#                 model, rays, z_coarse, coarse=True, sb=superbatch_size,
#             )
#             if self.vic_radius > 0:
#                 coarse_composite, coarse_composite_vic = coarse_composite
#
#             outputs = DotMap(
#                 coarse=self._format_outputs(
#                     coarse_composite, superbatch_size, want_weights=want_weights,
#                 ),
#             )
#
#             if self.vic_radius > 0:
#                 outputs.coarse.rgb_vic_mse = coarse_composite_vic[0]
#                 outputs.coarse.depth_vic_mse = coarse_composite_vic[1]
#                 outputs.coarse.flow_dist = coarse_composite_vic[2]
#
#
#             if self.using_fine:
#                 all_samps = [z_coarse]
#                 if self.n_fine - self.n_fine_depth > 0:
#                     all_samps.append(
#                         self.sample_fine(rays, coarse_composite[0].detach())
#                     )  # (B, Kf - Kfd)
#                 if self.n_fine_depth > 0:
#                     all_samps.append(
#                         self.sample_fine_depth(rays, coarse_composite[2])
#                     )  # (B, Kfd)
#                 z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
#                 z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
#                 fine_composite = self.composite(
#                     model, rays, z_combine_sorted, coarse=False, sb=superbatch_size,
#                 )
#
#                 if self.vic_radius > 0:
#                     fine_composite, fine_composite_vic = fine_composite
#
#
#                 outputs.fine = self._format_outputs(
#                     fine_composite, superbatch_size, want_weights=want_weights,
#                 )
#
#                 if self.vic_radius > 0:
#                     outputs.fine.rgb_vic_mse = fine_composite_vic[0]
#                     outputs.fine.depth_vic_mse = fine_composite_vic[1]
#                     outputs.fine.flow_dist = fine_composite_vic[2]
#
#             #print("finished rendering")
#             return outputs
#
#
#     def _format_outputs(
#         self, rendered_outputs, superbatch_size, want_weights=False,
#     ):
#         weights, rgb, depth, flow = rendered_outputs
#         if superbatch_size > 0:
#             rgb = rgb.reshape(superbatch_size, -1, 3)
#             depth = depth.reshape(superbatch_size, -1)
#             flow = flow.reshape(superbatch_size, -1, 3)
#             weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
#         ret_dict = DotMap(rgb=rgb, depth=depth, flow=flow)
#         if want_weights:
#             ret_dict.weights = weights
#         return ret_dict
#
#     def get_flow_diff(self, flow, flow_vic):
#         flow_diff = flow - flow_vic
#         flow_diff = torch.norm(flow_diff, dim=-1)
#         flow_diff = torch.mean(flow_diff)
#         return flow_diff