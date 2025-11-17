import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import LambdaLR
import datetime
from tqdm import tqdm
import cv2

from config import Config
from modeling.DGN import DGN

from utils.dataloader import get_im_gt_name_dict, create_dataloaders
from utils.transforms import Resize, Normalize, RandomHVFlip, RandomRotate
import utils.misc as misc
from utils.loss import dice_loss, sigmoid_ce_loss
import utils.metrics as metrics


def get_args_parser():
    parser = argparse.ArgumentParser('DualGazeNet Training', add_help=False)

     # Task configuration
    parser.add_argument("--mode", type=str, default='train',
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
                    default='L', 
                    type=str,
                    choices=['L', 'L*', 'B', 'B*', 'P', 'S'],
                    help='backbone type: L : (Hiera-L), ' \
                                        'L*: (pruned Hiera-L), ' \
                                        'B : (Hiera-B), ' \
                                        'B*: (pruned Hiera-L), ' \
                                        'P : (pvt_v2_b5), ' \
                                        'S : (swin_b), ' \
                                    )

    parser.add_argument("--input_size", type=int, default=512,
                        help="The size of input images")   
    
    # Training configuration
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument('--batch_size_train', default=8, type=int,
                    help='Batch size for training')

    parser.add_argument('--max_epoch_num', default=100, type=int, 
                        help='Maximum number of training epochs')
    
    parser.add_argument('--model_save_fre', default=3, type=int, 
                        help='Frequency (in epochs) for saving model checkpoints')
    
    parser.add_argument('--eval_interval', default=3, type=int,
                        help='Frequency (in epochs) for running evaluation')
    
    # Path configuration
    parser.add_argument("--output_dir", type=str, default='/root/autodl-tmp/train_output/DGN_L_new',
                        help="Path to the directory where checkpoints and records will be output")

    parser.add_argument("--resume_cpt", type=str,
                        default = '',
                        help="The path to the resume checkpoint")

    parser.add_argument("--visualize", type=str,
                        default = False,
                        help="")

    parser.add_argument("--pred_dir", type=str,
                        default = '',
                        help="Path to the directory where predictions will be saved")

    return parser.parse_args() 


class Trainer:
    def __init__(self, net, config, train_datasets=None, valid_datasets=None, args=None):
        self.mode = args.mode
        self.net = net
        self.train_datasets = train_datasets 
        self.valid_datasets = valid_datasets
        self.config = config
        self.args = args

        # Initialize all components
        self._set_seed(config.seed)
        self.net.to(self.args.device)    
        self._setup_dataloaders() 
        self._setup_optimizer()
        self._load_checkpoint()
        
        if args.mode == 'train':
            # Results files
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.test_results_file = os.path.join(args.output_dir, self.args.task, self.args.backbone, f"test_results_{timestamp}.txt")
            self.train_results_file = os.path.join(args.output_dir, self.args.task, self.args.backbone, f"train_results_{timestamp}.txt")
    
    def _set_seed(self, seed):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
    
    def _setup_dataloaders(self):
        """Initialize data loaders for training and validation"""   
        if self.mode == 'train':
            print("--- Creating training dataloaders ---")
            train_im_gt_list = get_im_gt_name_dict(self.train_datasets, flag="train")
            self.train_dataloaders, _ = create_dataloaders(
                train_im_gt_list,
                batch_size=self.args.batch_size_train,
                training=True,
                my_transforms=[
                    RandomHVFlip(prob=0.5), 
                    RandomRotate(prob=0.5),
                    Resize(input_size=self.args.input_size),
                    Normalize(),
                ],
            )
            print(f"{len(self.train_dataloaders)} training dataloaders created")

        print("--- Creating validation dataloaders ---")
        valid_im_gt_list = get_im_gt_name_dict(self.valid_datasets, flag="valid")
        self.valid_dataloaders, _ = create_dataloaders(
            valid_im_gt_list,
            batch_size=1,
            training=False,
            my_transforms=[
                Resize(input_size=self.args.input_size),
                Normalize(),
            ],
        )
        print(f"{len(self.valid_dataloaders)} validation dataloaders created")
    
    def _setup_optimizer(self):
        """Configure optimizer and learning rate scheduler"""
        print("--- Defining optimizer ---")
        self.optimizer = torch.optim.AdamW(
            [{"params": self.net.parameters(), "initial_lr": self.config.lr}], 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        
        def lr_lambda(epoch):
            # Linear warmup followed by exponential decay
            if epoch < self.config.warmup_epochs:
                return float(epoch + 1) / self.config.warmup_epochs
            else:
                return (0.98 ** (epoch - self.config.warmup_epochs))
        
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def _load_checkpoint(self):
        """Load checkpoint for resuming training"""
        if not self.args.resume_cpt:
            return
        
        checkpoint = torch.load(self.args.resume_cpt, map_location='cpu', weights_only=True)
        self.net.load_state_dict(checkpoint['net'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.args.start_epoch = checkpoint['epoch']
        print(f"Resumed training from checkpoint: {self.args.resume_cpt}, starting epoch: {self.args.start_epoch}")
    
    def train(self):
        """Main training loop"""
        os.makedirs(os.path.join(self.args.output_dir, self.args.task, self.args.backbone), exist_ok=True)
        epoch_start = getattr(self.args, 'start_epoch', 0)
        epoch_num = self.args.max_epoch_num
        
        self.net.train()
        
        for epoch in range(epoch_start, epoch_num):
            print(f"Epoch: {epoch}, Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Train for one epoch
            train_stats = self._train_epoch(epoch)
            self._record_train_results(epoch, train_stats)
            
            # Evaluate model
            if epoch % self.args.eval_interval == 0 or epoch == epoch_num - 1:

                test_stats = self.evaluate()
                self._record_test_results(epoch, test_stats)
                
                # Save model
                model_name = f"epoch_{epoch}.pth"
                model_path = os.path.join(self.args.output_dir, self.args.task, self.args.backbone, model_name)       
                checkpoint = self._create_checkpoint(epoch)
                torch.save(checkpoint, model_path)
                print(f'Saving model to: {model_path}')

                self.net.train()
        
        print("Training completed - reached maximum epoch number")
    
    def _train_epoch(self, epoch):
        """Train model for one epoch"""
        self.metric_logger = misc.MetricLogger(delimiter="  ")
        
        for data in self.metric_logger.log_every(self.train_dataloaders, 100):
            images, gts = data['image'], data['gt'] / 255.0
            images, gts = images.to(self.args.device), gts.to(self.config.device)
            
            # Forward pass
            masks = self.net(images)
            masks = F.interpolate(masks, [self.args.input_size, self.args.input_size], 
                                mode='bilinear', align_corners=False)
            
            # Calculate losses
            dice = dice_loss(masks, gts)
            bce = sigmoid_ce_loss(masks, gts)
            total_loss = bce + dice
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward() 
            self.optimizer.step()
            
            # Log losses
            loss_dict = {
                "loss": total_loss,
                "bce": bce,
                "dice": dice
            }
            self.metric_logger.update(**loss_dict)
        
        # Update learning rate
        self.lr_scheduler.step()
        
        print(f"Finished epoch: {epoch}")
        print("Averaged stats:", self.metric_logger)
        
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}
    
    def _record_train_results(self, epoch, train_stats):
        """Record training results to file"""
        write_mode = "w" if epoch == 0 else "a"
        with open(self.train_results_file, write_mode) as f:
            stats_details = "  ".join([f"{key}: {value:.4f}" for key, value in train_stats.items()])
            write_info = f"[epoch: {epoch}] {stats_details}\n"
            f.write(write_info)
            print(write_info)
    
    def _record_test_results(self, epoch, test_stats):
        """Record test results to file"""
        write_mode = "w" if epoch == 0 else "a"
        with open(self.test_results_file, write_mode) as f:
            stats_details = "  ".join([f"{key}: {value:.4f}" for key, value in test_stats.items()])
            write_info = f"[epoch: {epoch}] {stats_details}\n"
            f.write(write_info)
            print(write_info)
    
    def _create_checkpoint(self, epoch):
        """Create checkpoint dictionary"""
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(self.args)
        }
    
    def evaluate(self):
        """Evaluate model performance on validation datasets"""
        print("Validating...")
        
        sigmoid = torch.nn.Sigmoid()
        self.net.eval()
        
        test_stats = {}
        
        with torch.no_grad():
            score = 0
            for dataset_idx, valid_dataloader in enumerate(self.valid_dataloaders):
                dataset_name = self.valid_datasets[dataset_idx]['name']
                print(f"Start evaluating dataset: {dataset_name}")
                
                # Initialize evaluation metrics
                MAE = metrics.MAE()
                WFM = metrics.WeightedFmeasure()
                SM = metrics.Smeasure()
                EM = metrics.Emeasure()
                FM = metrics.Fmeasure()
                
                for data in tqdm(valid_dataloader, ncols=100):
                    name = data['name']
                    image = data['image'].to(self.args.device)
                    gt_ori = data['gt_ori'].squeeze(0)
                    
                    # Forward pass
                    mask = self.net(image)
                    out = sigmoid(mask)
                    out = F.interpolate(out, size=gt_ori.shape[-2:], 
                                      mode='bilinear', align_corners=False)
                    pred = (out * 255).squeeze().cpu().numpy().astype(np.uint8)
                    ori_label = gt_ori.cpu().numpy().astype(np.uint8)

                    if self.args.visualize:
                        os.makedirs(self.args.pred_dir, exist_ok=True)
                        cv2.imwrite(os.path.join('./pred', name + '.png'), pred)

                    FM.step(pred=pred, gt=ori_label)
                    WFM.step(pred=pred, gt=ori_label)
                    SM.step(pred=pred, gt=ori_label)
                    EM.step(pred=pred, gt=ori_label)
                    MAE.step(pred=pred, gt=ori_label)

                # fm = FM.get_results()["fm"]
                wfm = WFM.get_results()["wfm"]
                sm = SM.get_results()["sm"]
                em = EM.get_results()["em"]
                mae = MAE.get_results()["mae"]

                maxFm = FM.get_results()['mf']
                # meanFm = fm['curve'].mean()
                # fm_adp = fm['adp']
                em_mean = em['curve'].mean()
                em_max = em['curve'].max()

                test_stats.update({
                    f'mae_{dataset_name}': mae,
                    f'maxFm_{dataset_name}': maxFm,
                    f'wFm_{dataset_name}': wfm,
                    f'sm_{dataset_name}': sm,
                    f'em_mean_{dataset_name}': em_mean,
                    f'em_max_{dataset_name}': em_max,
                })

                score += (0.3 * (1 - mae) + 0.25 * (maxFm + wfm) + 0.2 * sm + 0.25 * (em_mean + em_max))

                print('============================')
                print(f'{dataset_name}:')
                print(f"MAE: {mae:.3f}, MaxF: {maxFm:.3f}, WeightF: {wfm:.3f}, SM: {sm:.3f}, meanEM: {em_mean:.3f}, maxEM: {em_max:.3f} \n")
            test_stats.update({
                    f'score': score
                })
        
        print("Validation complete")
        return test_stats


def main():

    config = Config()
    args = get_args_parser()
    print("Args: " + str(args) + '\n')
    
    train_datasets = config.datasets[args.task]['train']
    valid_datasets = config.datasets[args.task]['test']

    net = DGN(pretrained=True, backbone=args.backbone).to(args.device)

    trainer = Trainer(net=net, config=config,
                      train_datasets=train_datasets, 
                      valid_datasets=valid_datasets, 
                      args=args)
    trainer.train()


if __name__ == "__main__":
    main()