import argparse
import os
from pathlib import Path

import torch
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Resize
from torchvision.transforms.functional import normalize

from modeling.DGN import DGN


def parse_args():
    parser = argparse.ArgumentParser(description="DGN evaluation script.")
    parser.add_argument('--backbone', 
                    required=True, 
                    type=str,
                    choices=['L', 'L*', 'B', 'B*', 'P', 'S'],
                    help='backbone type: L : (Hiera-L), ' \
                                        'L*: (pruned Hiera-L), ' \
                                        'B : (Hiera-B), ' \
                                        'B*: (pruned Hiera-L), ' \
                                        'P : (pvt_v2_b5), ' \
                                        'S : (swin_b), ' \
                                    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default='cuda',
        help="Device to run inference on"
    )
    parser.add_argument(
        "--input_size", type=int, default=512,
        help="Input image size for the model"
    )
    parser.add_argument(
        "--im_dir", type=str, required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--pred_dir", type=str, default='./pred/inference_single',
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--image_ext", type=str, default=".jpg",
        help="Image file extension to process"
    )

    return parser.parse_args()


def image_preprocess(im, input_size):
    """Preprocess image for model inference"""
    # Normalization parameters
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(-1, 1, 1)
    
    # Normalize and resize
    im = normalize(im, mean, std)
    resize_transform = Resize((input_size, input_size))
    im = resize_transform(im.unsqueeze(0))
    
    return im


def get_image_paths(im_dir, image_ext=".jpg"):
    """Get all image paths from directory"""
    im_dir = Path(im_dir)
    image_paths = list(im_dir.glob(f"*{image_ext}"))
    image_paths.extend(list(im_dir.glob(f"*{image_ext.upper()}")))  # Handle uppercase extensions
    
    if not image_paths:
        raise ValueError(f"No images found in {im_dir} with extension {image_ext}")
    
    print(f"Found {len(image_paths)} images to process")
    return image_paths


def process_single_image(image_path, model, device, input_size):
    """Process a single image and return prediction"""
    # Read and preprocess image
    im_ori = cv2.imread(str(image_path))
    im_tensor = torch.tensor(
        cv2.cvtColor(im_ori, cv2.COLOR_BGR2RGB), 
        dtype=torch.float32, 
        device=device
    ).permute(2, 0, 1).contiguous()
    im_processed = image_preprocess(im_tensor, input_size)
    
    # Inference
    with torch.no_grad():
        mask = model(im_processed)
        out = torch.sigmoid(mask)
        
        # Resize to original dimensions
        out_resized = F.interpolate(
            out, 
            size=im_ori.shape[:2], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert to numpy array
        pred = (out_resized * 255).squeeze().cpu().numpy().astype(np.uint8)
    
    return pred


def main():
    args = parse_args()

    os.makedirs(args.pred_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    net = DGN(pretrained=False, backbone=args.backbone).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['net'], strict=True)
    net.eval()
    
    # Get image paths
    image_paths = get_image_paths(args.im_dir, args.image_ext)
    
    # Process images
    for image_path in image_paths:
        pred = process_single_image(image_path, net, args.device, args.input_size)
        output_path = os.path.join(args.pred_dir, image_path.name.replace(args.image_ext, '.png'))
        cv2.imwrite(output_path, pred)

    print(f"\nProcessing completed")


if __name__ == "__main__":
    main()