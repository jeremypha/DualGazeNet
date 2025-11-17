import argparse
import torch
from tqdm import tqdm
import os
import shutil
import numpy as np
import cv2
import os
import utils.metrics as metrics
from modeling.DGN import DGN
from utils.dataloader import get_im_gt_name_dict, create_dataloaders
from utils.transforms import Resize, Normalize, RandomHVFlip, RandomRotate
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import functional as F
import time


def get_args_parser():
    parser = argparse.ArgumentParser('Parser for Training', add_help=False)

    parser.add_argument("--checkpoint", type=str,
                        default = '/root/autodl-tmp/train_output/DGN_L_COD_2/epoch_81.pth',
                        help="The path to the whole checkpoint")

    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")
    
    parser.add_argument("--input_size", type=int, default=512,
                        help="The size of input images") 

    return parser.parse_args() 

def eval(net, dataloader, dataset):
    print(f"start eval dataset:{dataset}")

    save_dir = f'/root/autodl-tmp/COD_maps/DGN_L_new/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    # maeloss = nn.L1Loss()
    sigmoid = torch.nn.Sigmoid()

    MAE = metrics.MAE()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    FM = metrics.Fmeasure()

    with torch.no_grad():
        for data in tqdm(dataloader, ncols= 100):
            name = data['name'][0]
            image = data['image'].to(args.device)
            gt_ori = data['gt_ori'].squeeze(0)

            mask = net(image)
            out = sigmoid(mask)
            out = F.interpolate(out, size=gt_ori.shape[-2:], mode='bilinear', align_corners=False)
            pred = (out * 255).squeeze().cpu().data.numpy().astype(np.uint8)
            ori_label = gt_ori.cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir, name + '.png'), pred)
                
            FM.step(pred=pred, gt=ori_label)
            WFM.step(pred=pred, gt=ori_label)
            SM.step(pred=pred, gt=ori_label)
            EM.step(pred=pred, gt=ori_label)
            MAE.step(pred=pred, gt=ori_label)

    fm = FM.get_results()["fm"]
    pr = FM.get_results()["pr"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    maxFm = FM.get_results()['mf']
    # meanFm = fm['curve'].mean()
    # fm_adp = fm['adp']
    em_mean = em['curve'].mean()
    em_max = em['curve'].max()

    print("mae:{:.3f}, maxFm:{:.3f}, wfm:{:.3f}, sm:{:.3f}, meanEm:{:.3f}, maxEm:{:.3f}".format(mae, maxFm, wfm,sm, em_mean, em_max))


if __name__ == "__main__":

    dataset_duts_te = {"name": "DUTS-TE",
                "im_dir": "/root/autodl-tmp/dataset/DUTS/DUTS-TE/im",
                "gt_dir": "/root/autodl-tmp/dataset/DUTS/DUTS-TE/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png",}

    dataset_dut_omron = {"name": "DUT-OMRON",
                "im_dir": "/root/autodl-tmp/dataset/DUT-OMRON/im",
                "gt_dir": "/root/autodl-tmp/dataset/DUT-OMRON/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png",}

    dataset_ecssd_tr = {"name": "ECSSD-TR",
                    "im_dir": "/root/autodl-tmp/dataset/ECSSD/im",
                    "gt_dir": "/root/autodl-tmp/dataset/ECSSD/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}

    dataset_hkuis_te = {"name": "HKU-IS-TE",
                "im_dir": "/root/autodl-tmp/dataset/HKU-IS/im",
                "gt_dir": "/root/autodl-tmp/dataset/HKU-IS/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png",}
    
    dataset_pascals_te = {"name": "PASCAL-S",
                "im_dir": "/root/autodl-tmp/dataset/PASCAL-S/im",
                "gt_dir": "/root/autodl-tmp/dataset/PASCAL-S/gt",
                "im_ext": ".jpg",
                "gt_ext": ".png",}

    dataset_camo_te = {"name": "CAMO",
                    "im_dir": "/root/autodl-tmp/dataset/COD_TEsets/CAMO/im",
                    "gt_dir": "/root/autodl-tmp/dataset/COD_TEsets/CAMO/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}
    
    dataset_cod_10k_te = {"name": "COD10K",
                    "im_dir": "/root/autodl-tmp/dataset/COD_TEsets/COD10K/im",
                    "gt_dir": "/root/autodl-tmp/dataset/COD_TEsets/COD10K/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}

    dataset_nc4k_te = {"name": "NC4K",
                    "im_dir": "/root/autodl-tmp/dataset/COD_TEsets/NC4K/im",
                    "gt_dir": "/root/autodl-tmp/dataset/COD_TEsets/NC4K/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}
    
    dataset_chameleon_te = {"name": "CHAMELEON",
                    "im_dir": "/root/autodl-tmp/dataset/COD_TEsets/CHAMELEON/im",
                    "gt_dir": "/root/autodl-tmp/dataset/COD_TEsets/CHAMELEON/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}

    
    dataset_usod10k_te = {"name": "USOD10K-TE",
                    "im_dir": "/root/autodl-tmp/dataset/USOD10k/USOD10k/TE/RGB",
                    "gt_dir": "/root/autodl-tmp/dataset/USOD10k/USOD10k/TE/GT",
                    "im_ext": ".png",
                    "gt_ext": ".png",}

    dataset_usod_te = {"name": "USOD-TE",
                    "im_dir": "/root/autodl-tmp/dataset/USOD/images",
                    "gt_dir": "/root/autodl-tmp/dataset/USOD/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",}

    # valid_datasets = [dataset_duts_te, dataset_ecssd_tr, dataset_hkuis_te, dataset_pascals_te, dataset_dut_omron]
    valid_datasets = [dataset_camo_te, dataset_cod_10k_te, dataset_nc4k_te, dataset_chameleon_te]
    # valid_datasets = [dataset_cvc_300_te, dataset_cvc_clinic_te, dataset_cvc_colon_te, dataset_etis_te, dataset_kvasir_te]
    # valid_datasets = [dataset_usod10k_te]

    args = get_args_parser()
    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(
        valid_im_gt_list,
        batch_size=1, 
        training=False,
        my_transforms=[
            Resize(input_size=args.input_size),
            Normalize(),
        ],
        mode='valid'
    )
    print(len(valid_dataloaders), " valid dataloaders created")


    net = DGN().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = net.load_state_dict(checkpoint['net'], strict=False)

    for k, valid_dataloader in enumerate(valid_dataloaders):

        eval(net, valid_dataloader, valid_datasets[k]['name'])

        