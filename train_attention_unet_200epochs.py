import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os

# Import data loader and the NEW Attention U-Net model
from data_loader_unet import DRIVEDataset, train_transform
from model_attention_unet import AttentionUNet

# --- Configuration & Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 200
PATIENCE = 30
MIN_IMPROVEMENT = 0.001

TRAIN_IMG_DIR = "dataset/training/images/"
TRAIN_MASK_DIR = "dataset/training/1st_manual/"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu" 

# --- Custom Dice Loss Implementation ---
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# --- Metrics Calculation ---
def calculate_metrics(preds, targets):
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()
    
    TP = (preds * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    
    epsilon = 1e-8
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
    
    return accuracy.item(), sensitivity.item(), specificity.item(), dice.item()

# --- Core Training Function ---
def train_fn(loader, model, optimizer, bce_fn, dice_fn):
    loop = tqdm(loader, leave=True)
    epoch_loss, epoch_acc, epoch_sens, epoch_spec, epoch_dice = 0, 0, 0, 0, 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        predictions = model(data)
        
        bce_loss = bce_fn(predictions, targets)
        dice_loss = dice_fn(predictions, targets)
        loss = bce_loss + dice_loss

        acc, sens, spec, dice = calculate_metrics(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_sens += sens
        epoch_spec += spec
        epoch_dice += dice
        
        loop.set_postfix(loss=loss.item())
    
    n = len(loader)
    return epoch_loss/n, epoch_acc/n, epoch_sens/n, epoch_spec/n, epoch_dice/n

# --- Main Execution ---
def main():
    print(f"Initializing Attention U-Net 200-epoch training on: {DEVICE}")

    train_dataset = DRIVEDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize AttentionUNet
    model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
    bce_fn = nn.BCEWithLogitsLoss() 
    dice_fn = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    csv_file = "training_log.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Epoch", "Loss", "Accuracy", "Sensitivity", "Specificity", "Dice"])

    # Early Stopping Variables
    best_dice = 0.0
    patience_counter = 0
    best_model_path = "attention_unet_drive_200epochs_best.pth"

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        avg_loss, avg_acc, avg_sens, avg_spec, avg_dice = train_fn(train_loader, model, optimizer, bce_fn, dice_fn)
        
        print(f"Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | Acc: {avg_acc:.4f} | Sens: {avg_sens:.4f} | Spec: {avg_spec:.4f}")
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Logging the results for this epoch
            writer.writerow(["attention_unet_200epochs", epoch+1, avg_loss, avg_acc, avg_sens, avg_spec, avg_dice])

        # Early Stopping Logic
        if avg_dice > (best_dice + MIN_IMPROVEMENT):
            best_dice = avg_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"No sufficient improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! The model stopped making significant gains.")
            break

    print(f"\nTraining session complete! Best model weights saved to '{best_model_path}'.")

if __name__ == "__main__":
    main()
