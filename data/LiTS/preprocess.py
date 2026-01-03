import os
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

def preprocess_lits(data_root, output_root, window_min=-200, window_max=200):
    """
    Preprocesses LiTS dataset:
    1. Reads NIfTI volume and segmentation.
    2. Applies windowing to CT volume.
    3. Extracts slices and saves as PNG.
    """
    
    # Setup paths
    raw_data_dir = Path(data_root)
    # Assuming standard LiTS structure or flat directory. 
    # Adjust if volumes are in a different subfolder than segmentations.
    # Based on user description: "loaded LiTS dataset into data/LiTS folder"
    # and "segmentations" folder exists. 
    # We'll look for volumes in data_root and segmentations in data_root/segmentations
    
    vols_dir = raw_data_dir 
    segs_dir = raw_data_dir / "segmentations"
    
    # Check for volume_pt6 or other subfolders if volumes are not in root
    possible_vol_dirs = [raw_data_dir, raw_data_dir / "volumes", raw_data_dir / "volume_pt6"]
    
    
    # Note: If volumes are split across multiple folders, we might need to search all.
    # For now, let's keep the logic simple but robust to the user's current setup.

    
    out_img_dir = Path(output_root) / "images"
    out_lbl_dir = Path(output_root) / "labels"
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing LiTS data from {raw_data_dir} to {output_root}")
    
    # Iterate through potential patient IDs (0 to 130)
    # We use a loop because glob might miss paired files if naming is inconsistent,
    # and we want to enforce the ID structure.
    
    processed_counts = 0
    
    for pat_id in tqdm(range(131), desc="Processing Patients"):
        vol_name = f"volume-{pat_id}.nii"
        seg_name = f"segmentation-{pat_id}.nii"
        
        vol_path = vols_dir / vol_name
        seg_path = segs_dir / seg_name
        
        if not vol_path.exists():
            # Try looking in known subdirs
            found = False
            for d in possible_vol_dirs:
                temp_path = d / vol_name
                if temp_path.exists():
                    vol_path = temp_path
                    found = True
                    break
            
            if not found:
                # One last check: maybe recursively search? (expensive but useful)
                # matches = list(raw_data_dir.rglob(vol_name))
                # if matches:
                #     vol_path = matches[0]
                pass
            
        if not vol_path.exists() or not seg_path.exists():
            # print(f"Skipping Patient {pat_id}: Missing volume or segmentation")
            continue
            
        try:
            # Load NIfTI
            vol_nii = nib.load(str(vol_path))
            seg_nii = nib.load(str(seg_path))
            
            # Get data (H, W, D)
            vol_data = vol_nii.get_fdata()
            seg_data = seg_nii.get_fdata()
            
            # Ensure alignment
            if vol_data.shape != seg_data.shape:
                print(f"Shape mismatch for Patient {pat_id}: {vol_data.shape} vs {seg_data.shape}")
                continue
                
            # Iterate slices (Z-axis is usually the last dimension in NIfTI)
            num_slices = vol_data.shape[2]
            
            for i in range(num_slices):
                slice_vol = vol_data[:, :, i]
                slice_seg = seg_data[:, :, i]
                
                # Check if slice contains liver/tumor (classes 1 and 2)
                # If we want to filter empty slices to save space:
                if np.sum(slice_seg) == 0:
                    continue
                    
                # Windowing
                # Clip to [min, max]
                img = np.clip(slice_vol, window_min, window_max)
                # Normalize to [0, 255]
                img = (img - window_min) / (window_max - window_min) * 255.0
                img = img.astype(np.uint8)
                
                # Segmentation is already 0, 1, 2. 
                # 0: Background, 1: Liver, 2: Tumor
                # Save it as is (uint8). 
                # Note: Some viewers might show this as black, but values are there.
                mask = slice_seg.astype(np.uint8)
                
                # Rotate if necessary (NIfTI is often rotated 90 deg relative to standard image view)
                # Usually need to rotate 90 degrees counter-clockwise and flip
                img = np.rot90(img)
                mask = np.rot90(mask)
                img = np.fliplr(img)
                mask = np.fliplr(mask)

                
                # Save
                fname = f"vol_{pat_id}_slice_{i:03d}.png"
                
                cv2.imwrite(str(out_img_dir / fname), img)
                cv2.imwrite(str(out_lbl_dir / fname), mask)
                
                processed_counts += 1
                
        except Exception as e:
            print(f"Error processing Patient {pat_id}: {e}")
            
    print(f"Done! Processed {processed_counts} slices.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/LiTS", help="Path to LiTS folder containing volumes/ and segmentations/")
    parser.add_argument("--output_root", type=str, default="data/LiTS/LITS_PROCESSED", help="Output folder")
    args = parser.parse_args()
    
    preprocess_lits(args.data_root, args.output_root)
