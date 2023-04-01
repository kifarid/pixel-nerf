import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import tqdm
from datetime import datetime

from util import instantiate_from_config

def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=20,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-5.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="/work/dlclarge2/faridk-nerf_il/visuals",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="Directory to load checkpoints from",
    )
    parser.add_argument("--fps", type=int, default=15, help="FPS of video")
    return parser



args, conf = util.args.parse_args(extra_args)

#loading the model 
args.resume = True
device = util.get_cuda(args.gpu_id[0])
conf.model.params.ckpt_path = args.checkpoint
model = instantiate_from_config(conf.model).to(device).eval()
renderer = model.renderer
render_par = renderer.bind_parallel(model.net, args.gpu_id, simple_output=True).eval()


#create directory 
save_dir = os.path.join(args.visual_path, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")


dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)
data = dset[args.subset]

images = data["images"] if args.dataset_format != "rw" else data["images"][:, -1] # (NV, 3, H, W)
NV, _, H, W = images.shape
poses = data["poses"] if args.dataset_format != "rw" else data["poses"][:, -1] # (NV, 4, 4)
focal = data["focal"]
if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]
c = data.get("c")
if c is not None:
    c = c.to(device=device).unsqueeze(0)
# Get the distance from camera to origin
z_near = dset.z_near
z_far = dset.z_far

print("Generating rays")

radius = args.radius
angles =  torch.linspace(-3.14, 3.14, args.num_views + 1)[:-1]
shifts = torch.stack([torch.zeros_like(angles) , torch.sin(angles), torch.cos(angles)], dim=1) #
shifts = shifts * radius
#shifts = shifts.unsqueeze(1)
render_poses = poses[0].clone().unsqueeze(0).repeat(args.num_views, 1, 1)
render_poses[:, :3, -1] =  render_poses[:, :3, -1] + shifts

focal_render = focal if len(focal.shape) < 3 else focal[:, -1, :]
c_render = c
if c is not None:
    c_render = c if len(c.shape) < 3 else c[:, -1, :]

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal_render * args.scale,
    z_near,
    z_far,
    c=c_render * args.scale if c is not None else None,
).to(device=device) #.clone()
# (NV, H, W, 8)

focal = focal.to(device=device)


#if renderer.n_coarse < 64:
# Ensure decent sampling resolution
#renderer.n_coarse = 64
#renderer.n_fine = 128

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()
if random_source:
    src_view = torch.randint(0, NV, (1,))
else:
    src_view = source

src_focals = focal[:, src_view] if len(focal.shape)>2 else focal

if c is not None:
    src_cs = c[:, src_view] if len(c.shape)>2 else c
else:
    src_cs = None

with torch.no_grad():
    print("Encoding source view(s)")
    model.encode(
        images[src_view].unsqueeze(0), #.clone(),
        poses[src_view].unsqueeze(0), #.to(device=device).clone(),
        src_focals.clone(), #.clone(),
        c=src_cs #.clone(),
    )

print("Rendering", args.num_views * H * W, "rays")
vid_name = "{:04}".format(args.subset)
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name,  "video" + vid_name + timestamp  + ".mp4")


with torch.no_grad():
    
    all_rgb_fine = []
    for i, rays in enumerate(tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), H*W, dim=0) #args.ray_batch_size, dim=0)
    )):
        
        with torch.no_grad():
            print("Encoding source view(s)")
            data = dset[args.subset + 2*i]
            images = data["images"] if args.dataset_format != "rw" else data["images"][:, -1] # (NV, 3, H, W)
            NV, _, H, W = images.shape
            poses = data["poses"] if args.dataset_format != "rw" else data["poses"][:, -1] # (NV, 4, 4)

            model.encode(
                images[src_view].unsqueeze(0), #.clone(),
                poses[src_view].unsqueeze(0).to(device=device), #.clone(),
                src_focals.clone(),
                c=src_cs, #.clone(),
            )

        rgb, _depth = render_par(rays[None])
        
        #save image
        images_sv = data["images"][src_view, -1]  if len(data["images"].shape) == 5 else data["images"][src_view]
        img_np = (images_sv.permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.hstack((*img_np,))
        viewimg_path = os.path.join(
        args.visual_path, args.name, "video" + vid_name + timestamp + f"{i}" + "_view.jpg"
        )
        imageio.imwrite(viewimg_path, img_np)
        all_rgb_fine.append(rgb[0])
    
    
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    frames = rgb_fine.view(-1, H, W, 3)

print("Writing video")
imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)




print("Wrote to", vid_path, "view:", viewimg_path)
