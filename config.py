import os
from utils.loss import dice_loss, sigmoid_ce_loss

class Config():
    def __init__(self) -> None:

        # Train settings
        self.device = 'cuda' 
        self.seed = 42
        self.lr = 1e-4
        self.weight_decay = 1e-3
        self.warmup_epochs = 5

        # Model settings
        self.backbone = {
            'S': 'swin_b', 'P': 'pvt_v2_b5',
            'B*': 'hiera*_b', 'B': 'hiera_b', 
            'L*': 'hiera*_l', 'L': 'hiera_l',
        }
        self.dec_dims = {
            'swin_b': [128, 256, 512, 1024], 'pvt_v2_b5': [64, 128, 320, 512],

            'hiera*_b': [112, 224, 448], 'hiera*_l': [144, 288, 576],

            'hiera_b': [112, 224, 448, 896], 'hiera_l': [144, 288, 576, 1152],
        }

        # Pretrained weights
        self.weights_root_dir = './weights'
        model_name_to_weights_file = {
            'hiera*_b': 'sam2_hiera_base_plus.pt', 
            'hiera_b': 'sam2_hiera_base_plus.pt', 
            'hiera*_l': 'sam2_hiera_large.pt',
            'hiera_l': 'sam2_hiera_large.pt', 
            'swin_b': 'swin_base_patch4_window7_224.pth',
            'pvt_v2_b5': 'pvt_v2_b5_22k.pth'
        }

        self.weights = {}
        for model_name, weights_file in model_name_to_weights_file.items():
            self.weights[model_name] = os.path.join(self.weights_root_dir, weights_file)
        

        # DATASET settings
        datasets_root_dir = "./datasets"
        
        def create_dataset(task, name, im_ext=".jpg", gt_ext=".png"):
            return {
                "name": name,
                "im_dir": os.path.join(datasets_root_dir, task, name, 'im'),
                "gt_dir": os.path.join(datasets_root_dir, task, name, 'gt'),
                "im_ext": im_ext,
                "gt_ext": gt_ext,
            }

        self.datasets = {
            'SOD': {
                'train': [
                    create_dataset("SOD", "DUTS-TR"),
                ],
                'test': [
                    create_dataset("SOD", "DUTS-TE"),
                    create_dataset("SOD", "DUT-OMRON"),
                    create_dataset("SOD", "ECSSD"),
                    create_dataset("SOD", "HKU-IS"), 
                    create_dataset("SOD", "PASCAL-S"),
                ]
            },
            'COD': {
                'train': [
                    create_dataset("COD", "COD10K-TR"),
                    create_dataset("COD", "CAMO-TR"),
                ],
                'test': [
                    create_dataset("COD", "COD10K"),
                    create_dataset("COD", "CAMO"),
                    create_dataset("COD", "NC4K"),
                    create_dataset("COD", "CHAMELEON"),
                ]
            },
            'USOD': {
                'train': [
                    create_dataset("USOD", "USOD10K-TR", ".png"),
                ],
                'test': [
                    create_dataset("USOD", "USOD10K-TE", ".png"),
                ]
            },
        }
