import os
import glob
import nibabel as nib
from collections import Counter
import numpy as np

# Path to your clean images
DATA_DIR = "./data/processed/images"

def check_shapes():
    # 1. Find files
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))
    
    if not files:
        print(f"❌ No files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} images. Analyzing shapes...\n")
    
    # 2. Iterate and Record
    shapes = []
    
    # Header for the list
    print(f"{'Filename':<35} | {'Shape (X, Y, Z)':<20}")
    print("-" * 60)
    
    for f in files:
        try:
            # We only load the header (fast), not the whole image data
            img = nib.load(f)
            shape = img.shape
            shapes.append(shape)
            
            name = os.path.basename(f)
            print(f"{name:<35} | {str(shape):<20}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 3. Summary Statistics
    print("\n" + "="*30)
    print("       SHAPE SUMMARY")
    print("="*30)
    
    if not shapes:
        return

    # Break down dimensions
    widths  = [s[0] for s in shapes]
    heights = [s[1] for s in shapes]
    depths  = [s[2] for s in shapes]
    
    print(f"Total Images: {len(shapes)}")
    print(f"X-Axis (Width) : Min={min(widths)}, Max={max(widths)}, Avg={np.mean(widths):.1f}")
    print(f"Y-Axis (Height): Min={min(heights)}, Max={max(heights)}, Avg={np.mean(heights):.1f}")
    print(f"Z-Axis (Depth) : Min={min(depths)}, Max={max(depths)}, Avg={np.mean(depths):.1f}")
    
    # Check Divisibility by 16 (The U-Net Requirement)
    print("\n--- U-Net Safety Check ---")
    bad_shapes = 0
    for s in shapes:
        if s[0]%16 != 0 or s[1]%16 != 0 or s[2]%16 != 0:
            bad_shapes += 1
            
    if bad_shapes > 0:
        print(f"⚠️  WARNING: {bad_shapes} images are NOT divisible by 16.")
        print("   You MUST use 'DivisiblePadd' in your data loader, or U-Net will crash.")
    else:
        print("✅ Amazing! All images are divisible by 16 naturally.")

    # Most common shapes
    print("\nMost Common Shapes:")
    for shape, count in Counter(shapes).most_common(5):
        print(f"  {str(shape)}: {count} patients")

if __name__ == "__main__":
    check_shapes()