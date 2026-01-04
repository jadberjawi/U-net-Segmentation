import os
import shutil
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm # For a nice progress bar

# ---------------- CONFIGURATION ---------------- #
# Define where your current files are
RAW_DICOM_DIR = "./data/raw/DICOM"  # Folder with your .dcm files
RAW_MASK_DIR = "./data/raw/masks"    # Folder with your .nii.gz masks

# Define where you want the clean data to go
PROCESSED_DIR = "./data/processed"
# ----------------------------------------------- #

def preprocess_data():
    # 1. Setup Output Directories
    out_images_dir = os.path.join(PROCESSED_DIR, "images")
    out_labels_dir = os.path.join(PROCESSED_DIR, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # 2. Get List of DICOM files
    dicom_files = [f for f in os.listdir(RAW_DICOM_DIR) if f.endswith('.dcm')]
    
    print(f"Found {len(dicom_files)} DICOM files. Starting conversion...")

    for dcm_file in tqdm(dicom_files):
        # Extract the UID (remove .dcm)
        # file: 1.2.840...257.dcm  -> uid: 1.2.840...257
        uid = os.path.splitext(dcm_file)[0]
        
        # 3. Define Paths
        dcm_path = os.path.join(RAW_DICOM_DIR, dcm_file)
        
        # Look for the mask. User said it ends in "_mask.nii.gz"
        mask_filename = f"{uid}_mask.nii.gz"
        mask_src_path = os.path.join(RAW_MASK_DIR, mask_filename)
        
        # Check if matching mask exists
        if not os.path.exists(mask_src_path):
            print(f"Warning: No mask found for {dcm_file}. Skipping.")
            continue

        try:
            # 4. Convert DICOM to NIfTI
            # Read the single DICOM file
            image = sitk.ReadImage(dcm_path)
            
            # Define new clean filename (using the UID is fine)
            new_image_name = f"{uid}.nii.gz"
            new_mask_name = f"{uid}.nii.gz" # Same name, different folders
            
            # Save Image as NIfTI
            out_img_path = os.path.join(out_images_dir, new_image_name)
            sitk.WriteImage(image, out_img_path)
            
            # 5. Process Mask (Copy and Rename)
            # We assume the mask is already in correct orientation/spacing
            # If not, we might need to resample it to match the image here.
            # For now, we just copy it to the labels folder.
            out_mask_path = os.path.join(out_labels_dir, new_mask_name)
            shutil.copy(mask_src_path, out_mask_path)
            
        except Exception as e:
            print(f"Error processing {dcm_file}: {e}")

    print("Preprocessing complete!")
    print(f"Images saved to: {out_images_dir}")
    print(f"Labels saved to: {out_labels_dir}")

if __name__ == "__main__":
    preprocess_data()