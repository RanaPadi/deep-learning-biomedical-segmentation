# Retinal Vessel Segmentation: Standard U-Net vs. Attention U-Net

This repository contains the code, evaluation metrics, and comparative analysis for a biomedical image segmentation project conducted for the **Deep Learning Architectures** seminar at Ulm University.

The project focuses on extracting delicate blood vessel networks from color fundus images using deep learning, specifically comparing the performance and generalization capabilities of a Standard U-Net against an Attention U-Net on a highly constrained dataset.

## Project Objectives

1. **Core Objective:** Establish a robust training and evaluation pipeline using a Standard U-Net architecture on the DRIVE dataset.
2. **Secondary Objective:** Extend the architecture with Attention Gates (Attention U-Net) to analyze whether adding model complexity improves the segmentation of fine, capillary-level vascular structures.

## Dataset

This project uses the **DRIVE (Digital Retinal Images for Vessel Extraction)** dataset.

* **Training Set:** 20 color fundus images with corresponding manual expert masks.
* **Test Set:** 20 unseen images for evaluation.
* *Note: The dataset must be placed in the `dataset/` directory to run the scripts.*

## Repository Structure

```text
.
├── dataset/                              # DRIVE dataset (training/test images and masks)
├── evaluation_results/                   # Visual predictions, comparison images, and learning curves
├── data_loader_unet.py                   # PyTorch Dataset class and Albumentations pipeline
├── model_unet.py                         # Standard U-Net architecture implementation
├── model_attention_unet.py               # Attention U-Net architecture implementation
├── train_unet_200epochs.py               # Training loop, early stopping, and metric logging (U-Net)
├── train_attention_unet_200epochs.py     # Training loop, early stopping, and metric logging (Attention U-Net)
├── evaluate_unet_200epochs.py            # Evaluation and visualization script (U-Net)
├── evaluate_attention_unet_200epochs.py  # Evaluation and visualization script (Attention U-Net)
├── plot_learning_curves.py               # Generates Matplotlib graphs from the training log
├── training_log.csv                      # Epoch-by-epoch tracking of Loss, Accuracy, Sensitivity, Specificity, and Dice Score
└── README.md
```

## Installation & Requirements

The code is written in Python and uses PyTorch. To run the scripts locally, install the following dependencies:

```bash
# Optional but recommended: Create and activate a virtual environment
python -m venv unet_env
source unet_env/bin/activate  

# Install required packages
pip install -r requirements.txt
```

*Note: The scripts are currently configured to use Apple Silicon (`mps`). If you are running on a machine with an Nvidia GPU or CPU, PyTorch will automatically fall back to `cpu`, or you can manually update the `DEVICE` variable to `cuda`.*

## Usage

### 1. Training the Models

Both models are configured to train for up to 200 epochs using Binary Cross Entropy (BCE) + Dice Loss, and the AdamW optimizer. They include early stopping with a patience of 30 epochs.

```bash
# Train the Standard U-Net
python train_unet_200epochs.py

# Train the Attention U-Net
python train_attention_unet_200epochs.py
```

### 2. Evaluating the Models

The evaluation scripts run the trained weights against the 20 unseen test images, calculating the final F1/Dice score, Sensitivity, and Specificity. It also generates side-by-side `.png` comparisons.

```bash
# Evaluate Standard U-Net
python evaluate_unet_200epochs.py

# Evaluate Attention U-Net
python evaluate_attention_unet_200epochs.py
```

### 3. Visualizing Learning Curves

To plot the training metrics and visually analyze model convergence and overfitting:

```bash
python plot_learning_curves.py
```

## Key Findings & Results

The comparative study revealed a classic machine learning overfitting phenomenon due to the small size of the dataset (20 training images).

* **Standard U-Net (Test Dice: 0.7860):** The simpler baseline architecture generalized well, successfully capturing fine vascular structures with a high sensitivity of ~82.6%.
* **Attention U-Net (Test Dice: 0.7651):** The added complexity of the Attention Gates caused the model to severely overfit the training data (reaching a ~90% training Dice score), which resulted in slightly fragmented predictions on the unseen test set.

**Conclusion:** For highly constrained medical datasets, simpler architectures often yield more robust and generalizable segmentation models.
