
import argparse

from config import Config
from modeling.DGN import DGN
from train import Trainer


def get_args_parser():
    parser = argparse.ArgumentParser('DualGazeNet Testing', add_help=False)

    # Task configuration
    parser.add_argument("--mode", type=str, default='valid',
                        choices=['train', 'valid'],
                        help="Train or test")
    
    parser.add_argument('--task', 
                    default='SOD', 
                    type=str,
                    choices=['SOD', 'COD', 'USOD'],
                    help='Task type: SOD (Salient Object Detection), ' \
                                    'COD (Camouflaged Object Detection), ' \
                                    'USOD (Underwater Salient Object Detection)')
    
     # Model configuration
    parser.add_argument('--backbone', 
                    default='B', 
                    type=str,
                    choices=['L', 'L*', 'B', 'B*', 'P', 'S'],
                    help='backbone type: L : (Hiera-L), ' \
                                        'L*: (pruned Hiera-L), ' \
                                        'B : (Hiera-B), ' \
                                        'B*: (pruned Hiera-L), ' \
                                        'P : (pvt_v2_b5), ' \
                                        'S : (swin_b), ' \
                                    )

    parser.add_argument("--input_size", type=int, default=224,
                        help="The size of input images")   
    
    parser.add_argument("--device", type=str, default="cuda")

    # Path configuration
    parser.add_argument("--resume_cpt", type=str,
                        default = '/root/autodl-tmp/train_output/Final_Path/DGN_B_224.pth',
                        help="The path to the checkpoint")
    
    parser.add_argument('--visualize', default=False, type=bool,
                    help='Generate and save saliency map visualizations during inference')

    parser.add_argument("--pred_dir", type=str, default='./pred',
                        help="Path to the directory where predictions will be saved")

    return parser.parse_args() 


def main():

    config = Config()
    args = get_args_parser()
    print("Args: " + str(args) + '\n')

    valid_datasets = config.datasets[args.task]['test']

    net = DGN(pretrained=False, backbone=args.backbone).to(args.device)

    trainer = Trainer(net=net, config=config,
                      valid_datasets=valid_datasets, 
                      args=args)
    trainer.evaluate()


if __name__ == "__main__":
    main()