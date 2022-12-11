from .encoder import SpatialEncoder, ImageEncoder, FieldEncoder
from .resnetfc import ResnetFC
from torch import nn


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)

    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)

    elif mlp_type =="basic":
        # create a basic fc net with input size d_in + d_latent and output size d_out and n hidden layers
        d_out = conf.get("d_out", 4)
        n_hidden = conf.get("n_hidden", 2)
        d_hidden = conf.get("d_hidden", 128)
        # create a sequential model with n_hidden layers and d_hidden units each and a final layer with d_out units
        net = nn.Sequential(
            nn.Linear(d_in + d_latent, d_hidden),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU()) for _ in range(n_hidden)],
            nn.Linear(d_hidden, d_out)
        )

    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "field":
        net = FieldEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
