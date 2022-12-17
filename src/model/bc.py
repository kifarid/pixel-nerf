import numpy as np
import pytorch_lightning as pl
import torch
from dotmap import DotMap

import torch
import torchmetrics
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext

from src.util import instantiate_from_config
from src.model.model_util import make_mlp


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class BC(pl.LightningModule):
    # classic imitation learning policy

    def __init__(self,
                 model_config,
                 backbone_config,
                 scheduler_config=None,
                 ignore_keys=list(),
                 ckpt_path=None,
                 monitor=None,
                 base_learning_rate=1e-4,
                 train_backbone_bc_obj=False,
                 train_backbone_steps=0,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.train_backbone_steps = train_backbone_steps
        self.train_backbone_bc_obj = train_backbone_bc_obj
        self.instantiate_backbone(backbone_config)
        self.action_space = model_config["action_space"]
        self.scheduler_config = scheduler_config
        self.base_bc = make_mlp(model_config, d_in=self.backbone_model.net.d_latent)
        # .to(device=self.device)
        # create heads from action_space
        self.heads = nn.ModuleDict()
        for k, v in self.action_space.items():
            # create classification or regression one layer for each action in the action space
            self.heads[k] = nn.Linear(model_config.d_out, v["size"])

        # create losses needed
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if self.train_backbone_steps > 0:
            print("automatic optimization is dead")
            self.automatic_optimization = False
            self.backbone_model.automatic_optimization = False

        if self.no_grad_cond:#not self.train_backbone_bc_obj and self.train_backbone_steps <= 0:
            print("killing the gradient for the backbone")
            self.kill_backbone_training()

        # create torchmetrics accuracy for each discrete action
        self.train_acc = {}
        self.val_acc = {}
        # for k, v in self.action_space.items():
        #     if v["type"] == "discrete":
        #         task = 'binary' if v["size"] == 2 else 'multiclass'
        #         self.train_acc[k] = torchmetrics.Accuracy(task=task)
        #         self.val_acc[k] = torchmetrics.Accuracy(task=task)

    def calc_losses(self, gt, preds):

        loss_dict = {}
        for k, v in self.action_space.items():
            if v["type"] == "discrete":
                loss_dict[k] = self.cross_entropy_loss(preds[k], gt[k])
            elif v["type"] == "continuous":

                loss_dict[k] = self.mse_loss(preds[k], gt[k])

        # get mean of the losses
        loss = torch.mean(torch.stack(list(loss_dict.values())))
        # sum all losses in loss_dict
        #loss = sum(loss_dict.values())
        return loss, loss_dict

    def forward(self, data):

        x = self.get_input(data)
        x = self.base_bc(x)
        preds = {}
        for k, v in self.action_space.items():
            preds[k] = self.heads[k](x)
            # bound preds by range of action space
            if v["type"] == "continuous":
                preds[k] = torch.clamp(preds[k], v["min"], v["max"])

        return preds

    def step(self, data):
        with torch.no_grad():
            pred = self.forward(data)
            for k, v in self.action_space.items():
                #check if discrete then return max index along last dim
                if v["type"] == "discrete":
                    pred[k] = torch.argmax(pred[k], dim=-1)

        return pred

    def train_backbone(self, data, batch_idx):
        opt = self.backbone_model.optimizers()
        opt.zero_grad()
        loss = self.backbone_model.training_step(data, batch_idx)
        self.backbone_model.manual_backward(loss["loss"])
        opt.step()
        return loss

    def training_step(self, data, batch_idx):
        if self.train_backbone_steps > self.global_step:
            backbone_loss = self.train_backbone(data, batch_idx)
            self.log("backbone_loss", backbone_loss, on_step=True, on_epoch=True)

        if self.no_grad_cond:
            self.kill_backbone_training()

        preds = self.forward(data)
        # calculate accuracy metrics for discrete predictions
        # for k, v in self.action_space.items():
        #     if v["type"] == "discrete":
        #         self.train_acc[k](preds[k], data[k])

        loss, loss_dict = self.calc_losses(data["actions"], preds)
        # log loss and loss dict
        self.log("bc_train_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_train_loss_dict", loss_dict, on_step=True, on_epoch=True)



        # log accuracy metrics
        # for k, v in self.action_space.items():
        #     if v["type"] == "discrete":
        #         self.log(f"train_acc_{k}", self.train_acc[k].compute(), on_step=False, on_epoch=True)

        # check if automatic optimization is off and do backward pass manually
        if not self.automatic_optimization:
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().step()

        return loss # return loss to be used by the optimizer

    def validation_step(self, data, batch_idx):
        self.base_bc.eval()
        self.heads.eval()
        self.backbone_model.eval()
        with torch.no_grad():
            preds = self.forward(data)
            loss, loss_dict = self.calc_losses(data["actions"], preds)
            # for k, v in self.action_space.items():
            #     if v["type"] == "discrete":
            #         self.val_acc[k](preds[k], data[k])
        # log loss and loss dict
        self.log("bc_val_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_val_loss_dict", loss_dict, on_step=True, on_epoch=True)
        # for k, v in self.action_space.items():
        #     if v["type"] == "discrete":
        #         self.log(f"train_acc_{k}", self.val_acc[k].compute(), on_step=False, on_epoch=True)
        self.base_bc.train()
        self.heads.train()
        self.backbone_model.eval()
        if not self.no_grad_cond: #(self.train_backbone_steps > self.global_step) or self.train_backbone_bc_obj:
            self.backbone_model.train()
        return loss

    def test_step(self, data, batch_idx):
        self.base_bc.eval()
        self.heads.eval()
        self.backbone_model.eval()
        with torch.no_grad():
            preds = self.forward(data)
            loss, loss_dict = self.calc_losses(data["actions"], preds)
        # log loss and loss dict
        self.log("bc_test_loss", loss, on_step=True, on_epoch=True)
        self.log("bc_test_loss_dict", loss_dict, on_step=True, on_epoch=True)
        self.base_bc.train()
        self.heads.train()
        if not self.no_grad_cond:
            self.backbone_model.train()
        return loss

    def instantiate_backbone(self, config):
        model = instantiate_from_config(config)
        self.backbone_model = model
        if self.no_grad_cond: #(not self.train_backbone_bc_obj) and self.train_backbone_steps <= 0:
            self.kill_backbone_training()

    def kill_backbone_training(self):
        self.backbone_model.eval()
        self.backbone_model.train = disabled_train
        for param in self.backbone_model.parameters():
            param.requires_grad = False

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

        #with torch.no_grad():
        self.backbone_model.encode(images, poses, focal, c=c)
        latent = self.backbone_model.get_latent()
        return latent

    def get_input(self, batch):
        x = batch#.to(self.device)
        # x = x.float()#to(memory_format=torch.contiguous_format).float()

        all_images = x["images"].float()  # .to(device=device)  # (SB, NV, 3, H, W)
        SB, NV, _, H, W = all_images.shape
        all_poses = x["poses"].float()  # .to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = x.get("bbox").float()  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = x["focal"].float()  # (SB)
        all_c = x.get("c").float()  # (SB)

        with torch.no_grad() if not self.train_backbone_bc_obj else nullcontext():
            x = self.encode_backbone(all_images, all_poses, all_focals, c=all_c if all_c is not None else None)

        if self.no_grad_cond:
            x = x.detach()
        return x

    @property
    def no_grad_cond(self):
        return (self.train_backbone_steps <= self.global_step) and (not self.train_backbone_bc_obj)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.base_bc.parameters()) + list(self.heads.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        if self.scheduler_config is not None:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    # def training_epoch_end(self, outputs):
    #     self.train_acc.reset()

    # def validation_epoch_end(self, outputs):
    #     #self.log('valid_acc_epoch', self.valid_acc.compute())
    #     self.valid_acc.reset()