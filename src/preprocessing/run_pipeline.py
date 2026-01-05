from src.preprocessing import step1_convert

# Define your paths centrally here
RAW_DICOM_DIR = "./data/raw/DICOM"
RAW_MASK_DIR = "./data/raw/masks"
INTERIM_DIR = "./data/interim/01_converted"

def main():
    # Step 1: Raw -> Interim 1
    print("--- Step 1: Converting DICOM to NIfTI ---")
    step1_convert.run(
        input_dicom_dir="data/raw", 
        input_mask_dir="data/interim/01_converted",
        output_dir="data/interim/01_converted"
    )

    # Step 2: Interim 1 -> Interim 2
    # print("--- Step 2: Resampling to 1mm Isotropic ---")
    # step2_resample.run(input_dir="data/interim/01_converted", output_dir="data/interim/02_resampled")

    # Step 3: Interim 2 -> Processed
    # print("--- Step 3: Normalizing and Finalizing ---")
    # step3_normalize.run(input_dir="data/interim/02_resampled", output_dir="data/processed")

if __name__ == "__main__":
    main()