import torch
import torch.nn as nn
from collections import OrderedDict
from .pvt_v2 import pvt_v2_b5
from .swin_transformer import swin_transformer_B
from .hieradet import Hiera
from config import Config


config = Config()

def build_backbone(backbone_name, pretrained=True):

    if backbone_name == 'pvt_v2_b5':
        backbone = pvt_v2_b5()

    elif backbone_name == 'swin_b':
        backbone = swin_transformer_B(pretrained=False)

    elif backbone_name == 'hiera_l':
        backbone = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8]
        )

    elif backbone_name == 'hiera*_l':
        backbone = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36],
            global_att_blocks=[23, 33],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16],
            q_pool=2,
        )

    elif backbone_name == 'hiera_b':
        backbone = Hiera(
            embed_dim=112,
            num_heads=2,
        )


    elif backbone_name == 'hiera*_b':
        backbone = Hiera(
            embed_dim=112,
            num_heads=2,
            stages=[2, 3, 16],
            global_att_blocks=[12, 16],
            window_pos_embed_bkg_spatial_size=[14, 14],
            window_spec=[8, 4, 11],
            q_pool=2,
        )

    if pretrained:
        backbone = load_weights(backbone, backbone_name)
            
    return backbone


def load_weights(model, model_name):
    """Load pretrained weights for the model."""
    pretrained_dict = torch.load(config.weights[model_name], map_location='cpu', weights_only=True)

    if 'hiera' in model_name:
        model_state_dict = {}
        for key, value in pretrained_dict['model'].items():
            new_key = key.removeprefix('image_encoder.trunk.')
            model_state_dict[new_key] = value
            model.load_state_dict(model_state_dict, strict=False)

    elif 'pvt' in model_name:
        model.load_state_dict(pretrained_dict, strict=False)
    
    return model
