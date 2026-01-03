import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List
from PIL import Image

class LiTSDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.split = cfg.dataset.split # 'train', 'val', 'test'
        # Use dataset path if available, else global path
        if "paths" in cfg.dataset and "data_root" in cfg.dataset.paths:
             droot = cfg.dataset.paths.data_root
        else:
             droot = cfg.paths.data_root
             
        self.data_root = Path(droot) / "LITS_PROCESSED"
        
        self.img_dir = self.data_root / "images"
        self.lbl_dir = self.data_root / "labels"
        
        # Define Patient Splits
        # Train: 0-100
        # Val: 101-115
        # Test: 116-130
        if self.split == 'train':
            self.patient_ids = range(0, 101)
        elif self.split == 'val' or self.split == 'valid':
            self.patient_ids = range(101, 116)
        elif self.split == 'test':
            self.patient_ids = range(116, 131)
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        self.samples = []
        self._load_samples()
        
        print(f"LiTSDataset ({self.split}): Loaded {len(self.samples)} samples from {len(self.patient_ids)} patients.")

    def _load_samples(self):
        # Scan directory for files matching patient IDs
        # Filename format: vol_{id}_slice_{slice_idx}.png
        
        # To avoid scanning thousands of files every time, we can iterate through our patient IDs
        # and check for files. However, globbing is usually fast enough if we filter.
        
        # Efficient approach: List all files once, then filter.
        if not self.img_dir.exists():
            print(f"Warning: {self.img_dir} does not exist. Run preprocess.py first.")
            return

        all_files = sorted(list(self.img_dir.glob("*.png")))
        
        # Add tqdm for progress since we are opening files
        from tqdm import tqdm
        
        for fpath in tqdm(all_files, desc="Filtering Tumor Slices"):
            # Extract patient ID
            # vol_10_slice_005.png -> 10
            try:
                fname = fpath.name
                parts = fname.split('_')
                # parts: ['vol', '10', 'slice', '005.png']
                pid = int(parts[1])
                
                if pid in self.patient_ids:
                    # Check if mask has tumor (label 2)
                    lbl_path = self.lbl_dir / fname
                    mask = np.array(Image.open(lbl_path))
                    
                    # We are configuring dataset for TUMOR segmentation (label 2)
                    # Use prompt_type config check if strictly needed, but let's assume tumor logic for now.
                    if 2 in mask:
                         self.samples.append(fpath)
            except Exception:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        img_path = self.samples[idx]
        lbl_path = self.lbl_dir / img_path.name
        
        # Load Image (Grayscale but convert to RGB for SAM)
        img = np.array(Image.open(img_path).convert("RGB"))
        
        # Load Mask
        mask = np.array(Image.open(lbl_path)) # 0, 1, 2
        # Convert to binary for SAM (Foreground vs Background)
        # We can treat both Liver (1) and Tumor (2) as foreground, or just Tumor.
        # "Tumor Segmentation" implies Tumor. However, usually for LiTS we might want both. 
        # Since, we want to test in Live tumor segmentation. So, let's make it configurable or default to Tumor (2).
        # Let's do Tumor (2) -> 1, others -> 0.
        # If the mask is 0, 1, 2.
        
        # Let's default to Tumor (2) for now.
        mask = (mask == 2).astype(np.uint8)
        
        # Resize if needed
        if img.shape[0] != self.cfg.dataset.size:
            img = cv2.resize(img, (self.cfg.dataset.size, self.cfg.dataset.size))
            mask = cv2.resize(mask, (self.cfg.dataset.size, self.cfg.dataset.size), interpolation=cv2.INTER_NEAREST)
            
        # Generate Prompt
        prompt = self._generate_prompt(mask)
        
        return {
            "image": torch.tensor(img).permute(2, 0, 1).float(),
            "mask": torch.tensor(mask.astype(np.float32)),
            "prompt": prompt,
            "sample_id": img_path.stem
        }

    def _generate_prompt(self, mask: np.ndarray) -> Dict:
        h, w = mask.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return {"type": "none"}

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad = max(10, int(0.05 * w))

        box = [
            max(0, x1 - pad), max(0, y1 - pad),
            min(w, x2 + pad), min(h, y2 + pad)
        ]

        if self.cfg.dataset.prompt_type == "box":
            return {"type": "box", "box": box}
        elif self.cfg.dataset.prompt_type == "point":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return {"type": "point", "coord": [cx, cy], "label": 1}
        else:
            return {"type": "none"}
