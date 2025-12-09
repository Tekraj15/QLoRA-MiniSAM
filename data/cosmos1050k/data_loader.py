import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class COSMOSDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        split = cfg.dataset.split  # 'train', 'valid', 'test'
        modality = cfg.dataset.modality.lower()
        
        self.data_dir = Path(cfg.paths.data_root) / f"{split}_data"

        # Load all samples
        self.samples = []
        img_dir = self.data_dir / "images"
        label_dir = self.data_dir / "labels"

        for img_path in img_dir.glob("*.png"):
            if modality not in img_path.stem.lower():
                continue
            
            # Check for JSON first (legacy), then PNG (mask)
            label_path_json = label_dir / f"{img_path.stem}.json"
            label_path_png = label_dir / f"{img_path.name}" # Same filename as image
            
            if label_path_json.exists():
                self.samples.append({"image": str(img_path), "label": str(label_path_json), "type": "json"})
            elif label_path_png.exists():
                self.samples.append({"image": str(img_path), "label": str(label_path_png), "type": "png"})

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for modality '{modality}' in {split}_data")

        print(f"Loaded {len(self.samples)} samples from {split}_data ({modality})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        img = np.array(Image.open(sample["image"]).convert("RGB"))
        
        # Load Mask
        if sample["type"] == "json":
            with open(sample["label"]) as f:
                label = json.load(f)
            # Convert JSON to binary mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for shape in label.get("shapes", []):
                pts = np.array(shape["points"], np.int32)
                cv2.fillPoly(mask, [pts], 1)
        else: # PNG
            mask = np.array(Image.open(sample["label"]).convert("L"))
            mask = (mask > 0).astype(np.uint8) # Ensure binary 0/1

        # Resize if needed (SAM default: 1024)
        if img.shape[0] != self.cfg.dataset.size:
            img = cv2.resize(img, (self.cfg.dataset.size, self.cfg.dataset.size))
            mask = cv2.resize(mask, (self.cfg.dataset.size, self.cfg.dataset.size), interpolation=cv2.INTER_NEAREST)

        # Generate prompt
        prompt = self._generate_prompt(mask)

        return {
            "image": torch.tensor(img).permute(2, 0, 1).float(),
            "mask": torch.tensor(mask.astype(np.float32)),
            "prompt": prompt,
            "sample_id": Path(sample["image"]).stem
        }

    def _generate_prompt(self, mask: np.ndarray) -> Dict:
        h, w = mask.shape
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return {"type": "none"}

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad = max(10, int(0.05 * w))  # 5% padding

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