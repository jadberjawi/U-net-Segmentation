import os
import glob
from sklearn.model_selection import KFold
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, RandGaussianNoised, Rand3DElasticd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader, NibabelReader

def get_transforms(mode="train"):
    """
    Returns the transformation pipeline.
    mode: 'train' (augmentation enabled) or 'val' (only normalization)
    """
    # Base transforms applied to ALL data
    transforms = [
        LoadImaged(keys=["image", "label"], reader=NibabelReader),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]), # Normalize intensity to [0, 1]
        EnsureTyped(keys=["image", "label"]),
    ]

    # Augmentation only for training
    if mode == "train":
        transforms += [
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
            Rand3DElasticd(
                keys=["image", "label"], 
                sigma_range=(5, 8), 
                magnitude_range=(50, 150), 
                prob=0.2, 
                mode=("bilinear", "nearest") # Nearest for label to keep it 0/1
            ),
        ]
    
    return Compose(transforms)

def get_dataloaders(config):
    """
    Splits data into K-Folds and returns train/val dataloaders for the specified fold.
    """
    data_dir = config['data']['data_dir']
    
    # Assuming file naming: patient_001.nii.gz and patient_001_seg.nii.gz
    images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    
    # K-Fold Split
    kf = KFold(n_splits=config['data']['n_folds'], shuffle=True, random_state=42)
    splits = list(kf.split(data_dicts))
    train_idx, val_idx = splits[config['data']['train_fold']]
    
    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]

    # Create Datasets
    # CacheDataset accelerates training by caching preprocessed items in RAM
    train_ds = CacheDataset(data=train_files, transform=get_transforms("train"), cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=get_transforms("val"), cache_rate=1.0)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, # Validate one by one
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )

    return train_loader, val_loader