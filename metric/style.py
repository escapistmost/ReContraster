import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import argparse

# Configuration parameters
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RESULT_FILE = "style_statistics.txt"

# Data preprocessing
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# Custom collate function to filter invalid items
def custom_collate(batch):
    # Filter out invalid items
    valid_batch = [item for item in batch if item.get("valid", False)]
    
    if not valid_batch:
        # If there are no valid items, return an empty batch
        return {"valid_batch": False, "names": [item["name"] for item in batch]}
    
    # Use default collate function to process valid items
    collated_batch = {}
    for key in valid_batch[0].keys():
        if key != "valid":  # Skip valid flag
            try:
                collated_batch[key] = torch.utils.data._utils.collate.default_collate([item[key] for item in valid_batch])
            except Exception:
                # If a key cannot be collated, store it as a list
                collated_batch[key] = [item[key] for item in valid_batch]
    
    collated_batch["valid_batch"] = True
    return collated_batch

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        self.layer_mapping = {
            '3': "relu1_2",
            '8': "relu2_2", 
            '17': "relu3_4"
        }
        
        self.slices = nn.ModuleDict()
        current_slice = []
        for i in range(max(map(int, self.layer_mapping.keys())) + 1):
            current_slice.append(vgg[i])
            if str(i) in self.layer_mapping:
                self.slices[self.layer_mapping[str(i)]] = nn.Sequential(*current_slice)
                current_slice = []
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for layer_name in self.layer_mapping.values():
            x = self.slices[layer_name](x)
            features[layer_name] = x
        return features

def compute_style_gap(features, mask):
    layer_weights = {"relu1_2": 1.0, "relu2_2": 1.0, "relu3_4": 1.0}
    total_diff = 0.0
    
    for layer_name, feat in features.items():
        h, w = feat.shape[-2:]
        scaled_mask = nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest")
        
        # Split regions
        foreground = feat * scaled_mask
        background = feat * (1 - scaled_mask)
        
        # Calculate statistical differences
        diff = calculate_feature_diff(foreground, background)
        total_diff += layer_weights[layer_name] * diff
    
    return total_diff.item()

def calculate_feature_diff(fg, bg):
    fg_mean, fg_std = calc_stats(fg)
    bg_mean, bg_std = calc_stats(bg)
    
    mse = nn.MSELoss()
    return mse(fg_mean, bg_mean) + mse(fg_std, bg_std)

def calc_stats(x, eps=1e-6):
    n, c = x.shape[:2]
    x_flat = x.view(n, c, -1)
    
    valid_pixels = torch.sum(x_flat.abs() > eps, dim=2).clamp(min=1)
    mean = torch.sum(x_flat, dim=2) / valid_pixels
    var = torch.sum((x_flat - mean.unsqueeze(2))**2, dim=2) / valid_pixels
    
    return mean.view(n, c, 1, 1), torch.sqrt(var + eps).view(n, c, 1, 1)

def style_loss_caculate(img_path, mask_path):
    model = FeatureExtractor().to(DEVICE)
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image = image_transform(image)
    mask = mask_transform(mask).squeeze(0)
    image = image.to(DEVICE)
    mask = mask.unsqueeze(1).to(DEVICE)
    features = model(image)
    single_features = {k: v[0].unsqueeze(0) for k, v in features.items()}
    diff = compute_style_gap(single_features, mask)
    return diff

if __name__ == "__main__":
    print('请在调用 style_loss_caculate 时传入图片和 mask 路径')
