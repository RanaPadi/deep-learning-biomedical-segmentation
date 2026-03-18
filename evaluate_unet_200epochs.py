import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os

from data_loader_unet import DRIVEDataset
from model_unet import UNet
from train_unet_200epochs import calculate_metrics

# --- Configuration ---
TEST_IMG_DIR = "dataset/test/images/"
TEST_MASK_DIR = "dataset/test/1st_manual/"
MODEL_PATH = "unet_drive_200epochs_best.pth" # Points to the new best weights
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

test_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2()
])

def evaluate():
    print(f"Loading 200-epoch model on: {DEVICE}")
    test_dataset = DRIVEDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() 
    
    total_acc, total_sens, total_spec, total_dice = 0, 0, 0, 0
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for idx, (data, targets) in enumerate(test_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            predictions = model(data)
            acc, sens, spec, dice = calculate_metrics(predictions, targets)
            
            total_acc += acc
            total_sens += sens
            total_spec += spec
            total_dice += dice
            
            if idx < 3:
                img = data[0].cpu().permute(1, 2, 0).numpy()
                true_mask = targets[0].cpu().squeeze().numpy()
                pred_mask = torch.sigmoid(predictions[0]).cpu().squeeze().numpy()
                pred_mask = (pred_mask > 0.5).astype(np.float32)
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(img)
                ax[0].set_title("Input Image")
                ax[0].axis("off")
                
                ax[1].imshow(true_mask, cmap="gray")
                ax[1].set_title("Ground Truth Mask")
                ax[1].axis("off")
                
                ax[2].imshow(pred_mask, cmap="gray")
                ax[2].set_title(f"Standard U-Net Prediction (Dice: {dice:.3f})")
                ax[2].axis("off")
                
                # Updated image output name
                save_path = os.path.join(OUTPUT_DIR, f"result_comparison_unet_200epochs_{idx+1}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved visual comparison to {save_path}")

    n = len(test_loader)
    print("\n--- Final 200-Epoch Test Set Results ---")
    print(f"Average Accuracy:    {total_acc/n:.4f}")
    print(f"Average Sensitivity: {total_sens/n:.4f}")
    print(f"Average Specificity: {total_spec/n:.4f}")
    print(f"Average Dice Score:  {total_dice/n:.4f}")
    print(f"----------------------------------------")

if __name__ == "__main__":
    evaluate()