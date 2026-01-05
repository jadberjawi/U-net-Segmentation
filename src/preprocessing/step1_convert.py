import os
import shutil
import SimpleITK as sitk
from tqdm import tqdm

def run(input_dicom_dir, input_mask_dir, output_dir):
    """
    Step 1: Convert raw DICOMs to NIfTI and rename Masks to match.
    Saves to an interim folder (e.g., data/interim/01_converted).
    """
    
    # 1. Setup Output Directories inside the specific output_dir provided
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # 2. Get List of DICOM files
    # Check if input dir exists to avoid crash
    if not os.path.exists(input_dicom_dir):
        print(f"Error: Input directory {input_dicom_dir} does not exist.")
        return

    dicom_files = [f for f in os.listdir(input_dicom_dir) if f.endswith('.dcm')]
    
    print(f"--- Step 1: Found {len(dicom_files)} DICOM files. Converting... ---")

    for dcm_file in tqdm(dicom_files):
        # Extract UID
        uid = os.path.splitext(dcm_file)[0]
        
        # Define Source Paths
        dcm_path = os.path.join(input_dicom_dir, dcm_file)
        mask_filename = f"{uid}_mask.nii.gz"
        mask_src_path = os.path.join(input_mask_dir, mask_filename)
        
        # Check if matching mask exists
        if not os.path.exists(mask_src_path):
            print(f"Warning: No mask found for {dcm_file}. Skipping.")
            continue

        try:
            # 3. Convert DICOM to NIfTI
            image = sitk.ReadImage(dcm_path)
            
            # Define new filenames (Clean UID)
            new_image_name = f"{uid}.nii.gz"
            new_mask_name = f"{uid}.nii.gz" 
            
            # Save Image as NIfTI to the INTERIM folder
            out_img_path = os.path.join(out_images_dir, new_image_name)
            sitk.WriteImage(image, out_img_path)
            
            # 4. Copy Mask
            # NOTE: We assume mask geometry matches. 
            out_mask_path = os.path.join(out_labels_dir, new_mask_name)
            shutil.copy(mask_src_path, out_mask_path)
            
        except Exception as e:
            print(f"Error processing {dcm_file}: {e}")

    print(f"Step 1 Complete. Files saved to: {output_dir}")