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

from src.util import instantiate_from_config
from src.model.model_util import make_mlp


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class BC(pl.LightningModule):
    # classic imitation learning policy

    def __init__(self,
                 bc_config,
                 backbone_config,
                 scheduler_config=None,
                 ignore_keys=list(),
                 ckpt_path=None,
                 monitor=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.instantiate_backbone(backbone_config)
        self.action_space = bc_config["action_space"]
        self.scheudler_config = scheduler_config
        self.base_bc = make_mlp(bc_config, d_in=self.d_latent)
        # .to(device=self.device)
        # create heads from action_space
        self.heads = nn.ModuleDict()
        for k, v in self.action_space.items():
            # create classification or regression one layer for each action in the action space
            self.heads[k] = nn.Linear(self.base_bc.d_out, v["size"])

        # create losses needed
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        #self.train_acc = torchmetrics.Accuracy(task='multiclass')
        #self.valid_acc = torchmetrics.Accuracy(task='multiclass')

    def calc_losses(self, gt, preds):

        loss_dict = {}
        for k, v in self.action_space.items():
            if v["type"] == "discrete":
                loss_dict[k] = self.cross_entropy_loss(preds[k], gt[k])
            elif v["type"] == "continuous":
                loss_dict[k] = self.mse_loss(preds[k], gt[k])

        # sum all losses in loss_dict
        loss = sum(loss_dict.values())
        return loss, loss_dict

    def forward(self, data):
        x = self.get_input(data)
        x = self.base_bc(x)
        preds = {}
        for k, v in self.action_space.items():
            preds[k] = self.heads[k](x)
        # bound preds by range of action space
        for k, v in self.action_space.items():
            if v["type"] == "continuous":
                preds[k] = torch.clamp(preds[k], v["min"], v["max"])

        return preds

    def training_step(self, data, batch_idx):
        preds = self.forward(data)
        # calculate accuracy metrics for discrete predictions
        for k, v in self.action_space.items():
            if v["type"] == "discrete":
                self.train_acc(preds[k], data[k])

        loss, loss_dict = self.calc_losses(data["actions"], preds)
        # log loss and loss dict
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_loss_dict", loss_dict, on_step=True, on_epoch=True)
        #self.log('train_acc_epoch', self.train_acc.compute(), on_step=False, on_epoch=True)
        return loss # return loss to be used by the optimizer

    def validation_step(self, data, batch_idx):
        self.base_bc.eval()
        self.heads.eval()
        preds = self.forward(data)
        loss, loss_dict = self.calc_losses(data, preds)
        # log loss and loss dict
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_loss_dict", loss_dict, on_step=True, on_epoch=True)
        return loss

    def test_step(self, data, batch_idx):
        self.base_bc.eval()
        self.heads.eval()
        preds = self.forward(data)
        loss, loss_dict = self.calc_losses(data, preds)
        # log loss and loss dict
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_loss_dict", loss_dict, on_step=True, on_epoch=True)
        return loss

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.backbone_model = model.eval()
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
        with torch.no_grad():
            self.backbone_model.encode(images, poses, focal, c=c)
            latent = self.backbone_model.get_latent()
        return latent

    def get_input(self, batch):
        x = batch.to(self.device)
        # x = x.float()#to(memory_format=torch.contiguous_format).float()
        all_images = x["images"].float()  # .to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = x["poses"].float()  # .to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = x.get("bbox").float()  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = x["focal"].float()  # (SB)
        all_c = x.get("c").float()  # (SB)
        x = self.encode_backbone(all_images, all_poses, all_focals, c=all_c if all_c is not None else None)

        return x

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.backbone_trainable:
            print(f"{self.__class__.__name__}: Also optimizing backbone params!")
            params = params + list(self.backbone_model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
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

    def training_epoch_end(self, outputs):
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        self.valid_acc.reset()