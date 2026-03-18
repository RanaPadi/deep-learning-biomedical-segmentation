import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
LOG_FILE = "training_log.csv"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_curves():
    # 1. Load the data
    df = pd.read_csv(LOG_FILE)
    
    # Filter data for both models
    unet_df = df[df['Model'] == 'unet_200epochs']
    attn_df = df[df['Model'] == 'attention_unet_200epochs']
    
    # 2. Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Training Loss ---
    ax1.plot(unet_df['Epoch'], unet_df['Loss'], label='Standard U-Net', color='blue', linewidth=2)
    ax1.plot(attn_df['Epoch'], attn_df['Loss'], label='Attention U-Net', color='red', linewidth=2, linestyle='--')
    ax1.set_title("Training Loss Over 200 Epochs", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (BCE + Dice)", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # --- Plot 2: Training Dice Score ---
    ax2.plot(unet_df['Epoch'], unet_df['Dice'], label='Standard U-Net', color='blue', linewidth=2)
    ax2.plot(attn_df['Epoch'], attn_df['Dice'], label='Attention U-Net', color='red', linewidth=2, linestyle='--')
    ax2.set_title("Training Dice Score Over 200 Epochs", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Dice Score", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # 3. Save the plot
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "learning_curves_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Learning curves saved to: {save_path}")

if __name__ == "__main__":
    plot_curves()
