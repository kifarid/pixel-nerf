from .models import PixelNeRFNet, DNeRFNet


def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get("type", "pixelnerf")  # single
    if model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "dnerf":
        net = DNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
