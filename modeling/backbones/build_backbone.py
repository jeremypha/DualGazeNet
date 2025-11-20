import torch
import torch.nn as nn
import torch.nn.functional as F
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
        backbone = swin_transformer_B()

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

    elif 'swin' in model_name:
        pretrained_dict = pretrained_dict["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
        for k, v in list(pretrained_dict.items()):
            if ('attn.relative_position_index' in k) or ('attn_mask' in k):
                pretrained_dict.pop(k)
        if pretrained_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = pretrained_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.size()
            N2, C2, H, W = model.absolute_pos_embed.size()
            if N1 != N2 or C1 != C2 or L != H * W:
                    # logger.warning("Error in loading absolute_pos_embed, pass")
                print("no")
            else:
                pretrained_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

            # interpolate position bias table if needed
        relative_position_bias_table_keys = [k for k in pretrained_dict.keys() if
                                                 "relative_position_bias_table" in k]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = pretrained_dict[table_key]
            table_current = model.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if nH1 == nH2:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                            size=(S2, S2), mode='bicubic')
                    pretrained_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
        model.load_state_dict(pretrained_dict, strict=False)
        
    return model
