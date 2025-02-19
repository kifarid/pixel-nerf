import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

from src.util import instantiate_from_config
from src.model.model_util import make_mlp
from src.model.encoder import ImageEncoder


class Vanilla_BC(pl.LightningModule):
    # classic imitation learning policy

    def __init__(self,
                
                 model_config,
                 backbone_config,
                 scheduler_config=None,
                 ignore_keys=list(),
                 ckpt_path=None,
                 monitor=None,
                 discrete_loss_weight=5e-6,
                 base_learning_rate=1e-4,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_config  = scheduler_config
        self.instantiate_backbone(backbone_config)
        self.action_space = model_config["action_space"]
        self.base_bc = make_mlp(model_config, d_in=self.backbone_model.latent_size) # 3 for views
        self.heads = nn.ModuleDict()

        for k, v in self.action_space.items():
            # create classification or regression one layer for each action in the action space
            self.heads[k] = nn.Linear(model_config.d_out, v["size"])

        # create losses needed
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.discrete_loss_weight = discrete_loss_weight
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def calc_losses(self, gt, preds):

        loss_dict = {}
        for k, v in self.action_space.items():
            if v["type"] == "discrete":
                loss_dict[k] = self.discrete_loss_weight*self.cross_entropy_loss(preds[k], gt[k])
            elif v["type"] == "continuous":
                loss_dict[k] = self.mse_loss(preds[k], gt[k]) # removed sqrt

        # get mean of the losses
        loss = torch.mean(torch.stack(list(loss_dict.values())))
        # sum all losses in loss_dict
        # loss = sum(loss_dict.values())
        return loss, loss_dict

    def forward(self, data):

        x = self.get_input(data)
        x = self.base_bc(x)
        preds = {}
        for k, v in self.action_space.items():
            preds[k] = self.heads[k](x)
            # bound preds by range of action space
            #if v["type"] == "continuous":
            #    preds[k] = torch.clamp(preds[k], v["min"], v["max"])

        return preds

    def step(self, data):
        # move the data dict to the device of the model
        with torch.no_grad():
            data = {k: v.to(self.device) for k, v in data.items()}
            pred = self.forward(data)
            for k, v in self.action_space.items():
                # check if discrete then return max index along last dim
                if v["type"] == "discrete":
                    pred[k] = torch.argmax(pred[k], dim=-1)
        return pred

    def training_step(self, data, batch_idx):
        self.eval_all_models()
        preds = self.forward(data)
        loss, loss_dict = self.calc_losses(data["actions"], preds)
        # log loss and loss dict
        #print("\n loss in training step: ", loss, "\n")
        self.log("bc_train_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_train_loss_dict", loss_dict, on_step=True, on_epoch=True)
        self.eval_all_models()
        return loss  # return loss to be used by the optimizer

    def validation_step(self, data, batch_idx):
        self.eval_all_models()
        with torch.no_grad():
            preds = self.forward(data)
            loss, loss_dict = self.calc_losses(data["actions"], preds)
        #print("\n loss in validati step: ", loss, "\n")
        self.log("bc_val_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_val_loss_dict", loss_dict, on_step=True, on_epoch=True)
        self.eval_all_models()
        return loss

    def test_step(self, data, batch_idx):
        self.eval_all_models()
        with torch.no_grad():
            preds = self.forward(data)
            loss, loss_dict = self.calc_losses(data["actions"], preds)
        # log loss and loss dict
        self.log("bc_test_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_test_loss_dict", loss_dict, on_step=True, on_epoch=True)
        self.train_all_models()

        return loss

    def instantiate_backbone(self, config):
        model = ImageEncoder.from_conf(config)
        self.backbone_model = model
        # freeze backbone
        
    def eval_all_models(self):
        self.base_bc.eval()
        self.heads.eval()
        self.backbone_model.eval()

    def train_all_models(self):
        self.base_bc.train()
        self.heads.train()
        self.backbone_model.train()

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

    def encode_backbone(self, images, poses, focal, c=None):
        """
        Encode images using the backbone model.
        """
        # pass poses, focal and c to backbone
        num_objs = images.size(0)

        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            num_views_per_obj = 1

        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
        image_shape = torch.empty(2).to(images.device)
        image_shape[0] = images.shape[-1]
        image_shape[1] = images.shape[-2]

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
        focal = focal.float()
        focal[..., 1] *= -1.0

        if c is None:
            # Default principal point is center of image
            c = (image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))

        self.pass_to_backbone(poses=poses,
                              c=c,
                              num_views_per_obj=num_views_per_obj,
                              focal=focal,
                              image_shape=image_shape,
                              num_objs=num_objs)

        
        # run backbone withour gradients
        latent = self.backbone_model(images)
        
        self.pass_to_backbone(poses=None,
                              c=None,
                              num_views_per_obj=None,
                              focal=None,
                              image_shape=None,
                              num_objs=None)
        return latent

    def pass_to_backbone(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.backbone_model, k, v)

    def get_input(self, batch):
        x = batch
        all_images = x["images"].float()  # (SB, NV, 3, H, W)
        SB, NV, _, H, W = all_images.shape
        all_poses = x["poses"].float()  # (SB, NV, 4, 4
        all_focals = x["focal"].float()  # (SB)
        all_c = x.get("c").float()  # (SB)

        x = self.encode_backbone(all_images, all_poses, all_focals, c=all_c if all_c is not None else None)
        x = x.view(SB, NV, -1).mean(1) 
        return x

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.base_bc.parameters()) + list(self.heads.parameters()) + list(
            self.backbone_model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        if self.scheduler_config is not None:
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': StepLR(opt, **self.scheduler_config),
                    'interval': 'epoch',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
