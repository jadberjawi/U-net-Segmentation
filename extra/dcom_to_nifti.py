import os
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm


def dicom_to_nifti(dicom_root, output_dir, prefix="case"):
    """
    Convert each DICOM series folder to NIfTI and verify output by:
      - printing SITK size/spacing
      - reloading saved NIfTI with nibabel and printing shape
    """

    os.makedirs(output_dir, exist_ok=True)

    patient_dirs = sorted([
        d for d in os.listdir(dicom_root)
        if os.path.isdir(os.path.join(dicom_root, d))
    ])

    print(f"Found {len(patient_dirs)} DICOM folders\n")

    for idx, patient_id in enumerate(tqdm(patient_dirs)):
        patient_path = os.path.join(dicom_root, patient_id)

        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(patient_path)

            if not series_ids:
                print(f"⚠️ No DICOM series found in {patient_id}, skipping.")
                continue

            # take first series by default
            series_files = reader.GetGDCMSeriesFileNames(patient_path, series_ids[0])
            reader.SetFileNames(series_files)
            image = reader.Execute()

            # squeeze singleton 4th dim if present
            if image.GetDimension() == 4 and image.GetSize()[-1] == 1:
                image = image[:, :, :, 0]
                
            out_name = f"{prefix}_{idx:03d}.nii.gz"
            out_path = os.path.join(output_dir, out_name)
            sitk.WriteImage(image, out_path)

            # ---- SITK sanity ----
            size_xyz = image.GetSize()         # (X,Y,Z)
            spacing_xyz = image.GetSpacing()   # (sx,sy,sz)

            # ---- NIfTI sanity (what your ML pipeline will see) ----
            nii = nib.load(out_path)
            shape_nib = nii.shape              # usually (X,Y,Z) for 3D

            print(
                f"{out_name} | "
                f"SITK size XYZ={size_xyz} spacing={tuple(round(s, 4) for s in spacing_xyz)} | "
                f"nib shape={shape_nib}"
            )

        except Exception as e:
            print(f"❌ Error processing {patient_id}: {e}")

    print(f"\nDone. NIfTI files saved in {output_dir}")


if __name__ == "__main__":
    DICOM_ROOT = "data/test"
    OUTPUT_DIR = "data/test"

    dicom_to_nifti(DICOM_ROOT, OUTPUT_DIR, prefix="unseen")
