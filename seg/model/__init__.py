from .unet import UNet
from .transunet import TransUNet


def build_model_helper(model_name, model_cfg):
    return eval(f'{model_name}(model_cfg)')
