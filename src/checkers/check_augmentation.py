
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from monai.data import Dataset, DataLoader
from src.data_loader import get_transforms

# Define paths (adjust if your folder structure is different)
DATA_DIR = "./data/processed"

def visualize_augmentation():
    # 1. Get a single file
    images = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(DATA_DIR, "labels", "*.nii.gz")))
    
    if len(images) == 0:
        print("Error: No images found in data/processed/images/")
        return

    # Pick the first patient
    data_dict = [{"image": images[0], "label": labels[0]}]
    print(f"Checking augmentation on patient: {os.path.basename(images[0])}")

    # 2. Create two datasets
    # "Original" -> Uses validation transforms (No random rotation/elastic)
    ds_orig = Dataset(data=data_dict, transform=get_transforms("val"))
    
    # "Augmented" -> Uses train transforms (Includes random rotation/elastic)
    ds_aug = Dataset(data=data_dict, transform=get_transforms("train"))

    # 3. Load the data
    # We use batch_size=1 just to get the tensors out easily
    loader_orig = DataLoader(ds_orig, batch_size=1)
    loader_aug = DataLoader(ds_aug, batch_size=1)

    data_orig = next(iter(loader_orig))
    data_aug = next(iter(loader_aug))

    # 4. Extract Volumes (Batch index 0, Channel 0)
    vol_orig_img = data_orig["image"][0, 0]
    vol_orig_lbl = data_orig["label"][0, 0]
    
    vol_aug_img = data_aug["image"][0, 0]
    vol_aug_lbl = data_aug["label"][0, 0]

    # 5. Pick the middle slice (Z-axis)
    # Note: Augmented might have a different depth if you didn't resize, so we calculate mid-slice separately
    slice_idx_orig = vol_orig_img.shape[0] // 2
    slice_idx_aug = vol_aug_img.shape[0] // 2

    print(f"Original Shape: {vol_orig_img.shape}")
    print(f"Augmented Shape: {vol_aug_img.shape}")

    # 6. Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- Row 1: Original ---
    axes[0, 0].set_title("Original Image")
    axes[0, 0].imshow(vol_orig_img[slice_idx_orig, :, :], cmap="gray")
    axes[0, 0].axis('off')

    axes[0, 1].set_title("Original Mask")
    axes[0, 1].imshow(vol_orig_lbl[slice_idx_orig, :, :], cmap="jet")
    axes[0, 1].axis('off')

    # --- Row 2: Augmented ---
    axes[1, 0].set_title(f"Augmented Image\n(Rotated/Deformed)")
    axes[1, 0].imshow(vol_aug_img[slice_idx_aug, :, :], cmap="gray")
    axes[1, 0].axis('off')

    axes[1, 1].set_title("Augmented Mask")
    axes[1, 1].imshow(vol_aug_lbl[slice_idx_aug, :, :], cmap="jet")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_comparison.png")
    print("✅ Saved 'augmentation_comparison.png'. Check it out!")

if __name__ == "__main__":
    visualize_augmentation()