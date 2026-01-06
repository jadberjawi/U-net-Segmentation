import os
import yaml
import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, SpatialPadd,
    EnsureTyped, ScaleIntensityRangePercentilesd
)
from monai.data import Dataset, DataLoader, NibabelReader

from src.model import get_model
from src.utils import get_device  # or replace with: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_infer_transform():
    # MUST match your "val" transforms (no cropping, no augmentation)
    return Compose([
        LoadImaged(keys=["image"], reader=NibabelReader),
        EnsureChannelFirstd(keys=["image"]),
        SpatialPadd(keys=["image"], spatial_size=(80, 80, 80)),
        EnsureTyped(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1.0,
            upper=99.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            relative=False,
        ),
    ])


@torch.no_grad()
def infer(config_path, checkpoint_path, input_nii, output_mask_nii):
    # Load config (needed for model params)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device("auto") if "get_device" in globals() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = get_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Dataset/loader for single case
    ds = Dataset(data=[{"image": input_nii}], transform=get_infer_transform())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    x = batch["image"].to(device)  # (1, 1, 80, 80, 80)

    logits = model(x)              # (1, 2, 80, 80, 80)
    prob = torch.softmax(logits, dim=1)[:, 1]        # (1, 80, 80, 80)
    pred = (prob > 0.5).to(torch.uint8)[0]           # (80, 80, 80)

    # Save mask with same affine/header as input
    img = nib.load(input_nii)
    mask_nifti = nib.Nifti1Image(pred.cpu().numpy(), affine=img.affine, header=img.header)
    nib.save(mask_nifti, output_mask_nii)

    print("Saved mask to:", output_mask_nii)
    print("Pred foreground voxels:", int(pred.sum().item()))


if __name__ == "__main__":
    # Example usage (edit these paths)
    CONFIG = "configs/config.yaml"
    CKPT   = "models/best_model_fold0.pth"   # or your final model
    IN_NII = "data/test/unseen_000.nii.gz"
    OUT_NII = "data/test/predicted_mask"

    infer(CONFIG, CKPT, IN_NII, OUT_NII)
