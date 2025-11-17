## Data Loader
from __future__ import print_function, division

import numpy as np
from copy import deepcopy
from skimage import io
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


#### --------------------- Our DataLoader ---------------------####

def get_im_gt_name_dict(datasets: List[Dict[str, str]], flag: str = 'valid') -> List[Dict[str, Any]]:
    """Generate image and ground truth path dictionaries for datasets"""
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i, dataset in enumerate(datasets):
        print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", dataset["name"], "<<<---")
        
        # Get all image paths
        im_pattern = dataset["im_dir"] + os.sep + '*' + dataset["im_ext"]
        tmp_im_list = glob(im_pattern)
        print('-im-', dataset["name"], dataset["im_dir"], ': ', len(tmp_im_list))

        # Handle ground truth paths
        if not dataset["gt_dir"]:
            print('-gt-', dataset["name"], dataset["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            # Generate corresponding ground truth paths
            tmp_gt_list = [
                dataset["gt_dir"] + os.sep + 
                Path(x).name.replace(dataset["im_ext"], "") + dataset["gt_ext"] 
                for x in tmp_im_list
            ]
            # Filter out non-existent ground truth files
            tmp_gt_list = [gt_path for gt_path in tmp_gt_list if os.path.exists(gt_path)]
            print('-gt-', dataset["name"], dataset["gt_dir"], ': ', len(tmp_gt_list))

        name_im_gt_list.append({
            "dataset_name": dataset["name"],
            "im_path": tmp_im_list,
            "gt_path": tmp_gt_list,
            "im_ext": dataset["im_ext"],
            "gt_ext": dataset["gt_ext"]
        })
        
    return name_im_gt_list


def create_dataloaders(
    name_im_gt_list: List[Dict[str, Any]], 
    my_transforms: Optional[List] = None, 
    batch_size: int = 1, 
    training: bool = False, 
) -> Tuple[List[DataLoader], List[Dataset]]:
    """Create dataloaders for training or validation"""
    if my_transforms is None:
        my_transforms = []
        
    gos_dataloaders = []
    gos_datasets = []

    if not name_im_gt_list:
        return gos_dataloaders, gos_datasets

    num_workers_ = _get_optimal_num_workers(batch_size)

    if training:
        # Training mode - concatenate all datasets
        for dataset_config in name_im_gt_list:
            gos_dataset = OurDataset(
                [dataset_config], 
                transform=transforms.Compose(my_transforms), 
                training = training,
            )
            gos_datasets.append(gos_dataset)

        combined_dataset = ConcatDataset(gos_datasets)

        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=num_workers_,
            drop_last=True,  # Drop incomplete batch
        )
        gos_dataloaders = dataloader
        gos_datasets = combined_dataset

    else:
        # Validation/Test mode - keep datasets separate
        for dataset_config in name_im_gt_list:
            gos_dataset = OurDataset(
                [dataset_config],
                transform=transforms.Compose(my_transforms),
                training = training,
            )
            
            dataloader = DataLoader(
                gos_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffle for validation
                num_workers=num_workers_,
                drop_last=False,  # Keep all samples
            )
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets


def _get_optimal_num_workers(batch_size: int) -> int:
    """Determine optimal number of workers based on batch size"""
    if batch_size <= 1:
        return 1
    elif batch_size <= 4:
        return 2
    elif batch_size <= 8:
        return 4
    else:
        return 8


class OurDataset(Dataset):
    def __init__(
        self, 
        name_im_gt_list: List[Dict[str, Any]], 
        transform: Optional[Any] = None, 
        training :bool = False,
    ):
        self.transform = transform
        self.dataset = self._build_dataset(name_im_gt_list)
        self._length = len(self.dataset["im_path"])
        self.training = training
    
    def _build_dataset(self, name_im_gt_list: List[Dict[str, Any]]) -> Dict[str, List]:
        """Build unified dataset dictionary from multiple configurations"""
        dataset = {
            "data_name": [],
            "im_name": [],
            "im_path": [],
            "ori_im_path": [],
            "gt_path": [],
            "ori_gt_path": [],
            "im_ext": [],
            "gt_ext": []
        }
        
        for dataset_config in name_im_gt_list:
            dataset_name = dataset_config["dataset_name"]
            im_paths = dataset_config["im_path"]
            gt_paths = dataset_config["gt_path"]
            im_ext = dataset_config["im_ext"]
            gt_ext = dataset_config["gt_ext"]
            
            # Skip if no images available
            if not im_paths:
                continue
                
            # Process dataset in batch
            n_samples = len(im_paths)
            dataset["data_name"].extend([dataset_name] * n_samples)
            dataset["im_path"].extend(im_paths)
            dataset["ori_im_path"].extend(deepcopy(im_paths))
            dataset["gt_path"].extend(gt_paths)
            dataset["ori_gt_path"].extend(deepcopy(gt_paths))
            dataset["im_ext"].extend([im_ext] * n_samples)
            dataset["gt_ext"].extend([gt_ext] * n_samples)
            
            for im_path in im_paths:
                im_name = Path(im_path).stem  # Get filename without extension
                dataset["im_name"].append(im_name)
        
        return dataset
    
    def __len__(self) -> int:
        """Return total number of samples in dataset"""
        return self._length
    
    def _load_image(self, im_path: str) -> torch.Tensor:
        """Load and preprocess image tensor"""
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Image file not found: {im_path}")
            
        im = cv2.imread(im_path)
        if im is None:
            raise ValueError(f"Failed to load image: {im_path}")
        
        # Convert BGR to RGB and adjust tensor dimensions
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = torch.tensor(im_rgb, dtype=torch.float32)
        im_tensor = im_tensor.permute(2, 0, 1).contiguous()
        
        return im_tensor
    
    def _load_gt_mask(self, gt_path: str) -> torch.Tensor:
        """Load and preprocess ground truth mask"""
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
            
        gt_ori = io.imread(gt_path, as_gray=True)
        
        # Handle multi-channel images
        if gt_ori.ndim > 2:
            gt_ori = gt_ori[:, :, 0]
        
        # Normalize to 0-255 range
        gt_max = gt_ori.max()
        if gt_max <= 1.0:
            gt_ori = (gt_ori * 255).astype(np.uint8)
        elif gt_max > 255:
            gt_ori = (gt_ori / gt_max * 255).astype(np.uint8)
        
        return torch.from_numpy(gt_ori)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data sample by index"""
        name = self.dataset["im_name"][idx]
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        
        im = self._load_image(im_path)
        gt_ori = self._load_gt_mask(gt_path)
        
        # Build sample dictionary
        sample = {
            "name": name,
            "image": im,
            "gt": gt_ori,

        }

        if not self.training:
            sample["gt_ori"] = gt_ori

        if self.transform:
            sample = self.transform(sample)
            
        return sample


if __name__ == '__main__':

    from config import Config
    from transforms import Resize, Normalize, RandomHVFlip, RandomRotate
    config = Config()

    train_datasets = config.datasets['SOD']['train']
    valid_datasets = config.datasets['SOD']['test']
    print("--- create training dataloader ---")
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloaders, train_datasets = create_dataloaders(
        train_im_gt_list,
        batch_size=config.batch_size_train,
        training=True,
        my_transforms=[
            RandomHVFlip(prob=0.5), 
            RandomRotate(prob=0.5),
            Resize(input_size=512),
            Normalize(),
        ],
        mode='train',
    )
    print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(
        valid_im_gt_list,
        batch_size=config.batch_size_valid,
        training=False,
        my_transforms=[
            Resize(input_size=512),
            Normalize(),
        ],
    mode='valid',
    )
    print(len(valid_dataloaders), " valid dataloaders created")    
