import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class VolEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """

    def __init__(
        self,
        shape_in=(128, 64, 64, 64),
        num_conv_layers=3,
        num_mlp_layers=3,
        d_latent=300,
        use_leaky_relu=True,
    ):
        super().__init__()
        dim_in = shape_in[0]
        self.act = nn.LeakyReLU if use_leaky_relu else nn.ReLU

        self.conv3_layers = [
            nn.Conv3d(dim_in, 128, 3, padding=1),
            self.act(),
            nn.Conv3d(128, 128, 3, stride=2, padding=1),
            self.act(),
            nn.Conv3d(128, 64, 3, stride=2, padding=1),
        ]
        self.conv_model = nn.Sequential(*self.conv3_layers)
        mlp_dim_in = 64*math.prod(shape_in[1:])//(4**3)
        self.mlp_layers = [
            nn.Linear(mlp_dim_in, 300),
            self.act(),
            nn.Linear(300, 300),
            self.act(),
            nn.Linear(300, d_latent),
        ]
        self.mlp_model = nn.Sequential(*self.mlp_layers)

    def forward(self, x):

        x = self.conv_model(x)
        x = x.flatten(1, -1)
        x = self.mlp_model(x)

        return x
