import torch
import matplotlib.pyplot as plt
import os
from src.data_loader import get_dataloaders
from src.model import get_model
from monai.transforms import AsDiscrete

# 1. Setup - UPDATED CONFIG
config = {
    'data': {
        'data_dir': './data/processed',  # <--- THIS WAS MISSING
        'batch_size': 1, 
        'train_fold': 0,
        'n_folds': 5,
        'num_workers': 4
    },
    'model': {'in_channels': 1, 'out_channels': 2},
    'training': {'val_interval': 1}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model & Data
print("Loading data...")
_, val_loader = get_dataloaders(config)
model = get_model(config).to(device)

# Load the weights
model_path = "models/best_model_fold0.pth" 
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device)) # map_location handles CPU/GPU mismatch
    print(f"Loaded weights from {model_path}")
else:
    print("⚠️  No trained weights found! Using random weights.")

model.eval()

# 3. Predict on one patient
# We loop until we find a patient that actually has a mask (some might be empty)
print("Finding a patient with a visible mask...")
found_valid = False
input_img, label_img, pred_img = None, None, None

with torch.no_grad():
    for i, data in enumerate(val_loader):
        inputs, labels = data["image"].to(device), data["label"].to(device)
        
        # Check if this patient actually has a mask (sum > 0)
        if labels.sum() > 0:
            outputs = model(inputs)
            preds = AsDiscrete(argmax=True)(outputs)
            
            # Find the slice with the biggest organ area
            # Sum across H and W to find the 'deepest' slice index
            slice_sums = labels[0, 0].sum(dim=(0, 1))
            slice_idx = torch.argmax(slice_sums).item()
            
            # Grab that slice for plotting
            input_img = inputs[0, 0, :, :, slice_idx].cpu().numpy()
            label_img = labels[0, 0, :, :, slice_idx].cpu().numpy()
            pred_img = preds[0, 0, :, :, slice_idx].cpu().numpy()
            
            print(f"Found interesting slice at Z-index: {slice_idx}")
            found_valid = True
            break
            
    if not found_valid:
        print("Warning: Could not find any patient with a mask in the validation set!")

# 4. Visualization
if found_valid:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_img, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(label_img, cmap="jet", interpolation="nearest")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Model Prediction")
    plt.imshow(pred_img, cmap="jet", interpolation="nearest")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("debug_prediction.png")
    print("✅ Saved 'debug_prediction.png'. Open it to diagnose the model!")