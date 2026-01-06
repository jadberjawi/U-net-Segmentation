import nibabel as nib
import numpy as np
import glob

# Check the first few labels
files = glob.glob("./data/processed/labels/*.nii.gz")[:3]
for f in files:
    data = nib.load(f).get_fdata()
    print(f"{f}: Unique values = {np.unique(data)}")