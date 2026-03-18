import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DRIVEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List all image files
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
      
        # Load images as numpy arrays (required by Albumentations)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Normalize mask to binary (0.0 and 1.0)
        mask[mask > 0.0] = 1.0

        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            # Add channel dimension to mask (1, H, W) for PyTorch
            mask = mask.unsqueeze(0) 

        return image, mask

# Define the augmentations
train_transform = A.Compose([
    # 1. Resize to a square (fixes rotation batching and prepares for U-Net)
    A.Resize(height=512, width=512),
    
    # 2. Geometric augmentations
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # 3. Elastic deformation (removed outdated alpha_affine parameter)
    A.ElasticTransform(alpha=1, sigma=50, p=0.5), 
    
    # 4. Normalization and conversion to PyTorch Tensor
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])

# Example of how to load the data
if __name__ == "__main__":
    TRAIN_IMG_DIR = "dataset/training/images/"
    TRAIN_MASK_DIR = "dataset/training/1st_manual/"
    
    train_dataset = DRIVEDataset(
        image_dir=TRAIN_IMG_DIR, 
        mask_dir=TRAIN_MASK_DIR, 
        transform=train_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Test a single batch
    images, masks = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
