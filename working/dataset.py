# src/dataset.py

import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from .radar_utils import preprocess_frame # Import from the new utility file
from matplotlib import cm

# class RadarFrameDatasetOld(Dataset):

# def __init__(self, h5_path, stats_path, processor, is_train=False,):

# self.h5_path = h5_path

# self.processor = processor

# self.is_train = is_train

# self.data_file = h5py.File(self.h5_path, 'r')

# self.rdm_frames = self.data_file['data']

# self.labels = self.data_file['labels']



# with open(stats_path, 'r') as f:

# stats = json.load(f)

# self.min_val, self.max_val = stats['min'], stats['max']



# if self.is_train:

# self.augmentation_transform = transforms.Compose([

# transforms.RandomHorizontalFlip(p=0.5),

# transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),

# transforms.ColorJitter(brightness=0.2, contrast=0.2),

# ])



# def __len__(self):

# return len(self.labels)



# def __getitem__(self, idx):

# db_frame = self.rdm_frames[idx]

# label = self.labels[idx]



# # Use the shared utility function to create the base PIL Image

# # image = preprocess_frame(db_frame, self.min_val, self.max_val, f"train_seen/{idx}.png")

# image = preprocess_frame(db_frame, self.min_val, self.max_val, f"test_seen/{idx}.png")

# # image = preprocess_frame(db_frame, self.min_val, self.max_val, f"test_r1_seen/{idx}.png")



# if self.is_train:

# image = self.augmentation_transform(image)

# processed = self.processor(images=image, return_tensors="pt", padding=True)


# return {

# 'pixel_values': processed['pixel_values'].squeeze(0),

# 'label': torch.tensor(label, dtype=torch.long)

# }





# --- RadarFrameDataset (from your previous code) ---
class RadarFrameDataset(Dataset):
    """
    Loads pre-processed RD heatmaps and prepares them for a model.
    It now accepts a transform pipeline for data augmentation.
    """

    def __init__(self, h5_path, stats_path, processor=None, transform=None):

        self.h5_path = h5_path
        self.processor = processor
        self.transform = transform

        with open(stats_path, 'r') as f:
            stats = json.load(f)
            self.min_val, self.max_val = stats['min'], stats['max']
            self.cmap = cm.get_cmap('viridis')

        with h5py.File(self.h5_path, 'r') as hf:    
            self.length = len(hf['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            rd_map = hf['data'][idx]
            label = hf['labels'][idx]

        # --- Standard Preprocessing ---
        rd_map_clipped = np.clip(rd_map, self.min_val, self.max_val)
        denominator = self.max_val - self.min_val
        if denominator < 1e-6: denominator = 1e-6
        rd_map_normalized = (rd_map_clipped - self.min_val) / denominator
        colored_map = self.cmap(rd_map_normalized)
        rgb_image_array = (colored_map[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb_image_array)

        # --- Apply Transforms ---
        # The CLIP processor handles basic transforms like resize and normalize.
        # The custom augmentation transform pipeline is applied after.
        if self.processor:
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed['pixel_values'].squeeze(0)
        else:
            pixel_values = transforms.ToTensor()(image) # Fallback if no processor

        if self.transform:
            pixel_values = self.transform(pixel_values)
        
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}