import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

RAW_DIR = Path("data/cosmo1050k/raw_data")
PREP_DIR = Path("data/cosmo1050k/preprocessed_data")
PREP_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (1024, 1024)  # SAM default; use (256, 256) for faster testing
IMG_EXT = ".png"

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] and convert to uint8."""
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

def process_sample(img_path: Path, mask_path: Path, sample_id: str):
    # Load image
    img = np.array(Image.open(img_path).convert("RGB"))
    if img.shape[:2] != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    # Load mask (assume binary or multi-class JSON)
    with open(mask_path) as f:
        mask_data = json.load(f)
    # Convert polygon/bbox to binary mask (simplified)
    mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    for shape in mask_data.get("shapes", []):
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)

    # Save
    img_save_path = PREP_DIR / f"{sample_id}_img{IMG_EXT}"
    mask_save_path = PREP_DIR / f"{sample_id}_mask{IMG_EXT}"
    Image.fromarray(normalize_image(img)).save(img_save_path)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_save_path)

    return {"image": str(img_save_path), "mask": str(mask_save_path)}

def main():
    print(f"Preprocessing COSMOS raw data from {RAW_DIR} → {PREP_DIR}")
    samples = []
    img_files = list((RAW_DIR / "images").glob("*.jpg")) + list((RAW_DIR / "images").glob("*.png"))
    
    for img_path in tqdm(img_files, desc="Preprocessing"):
        sample_id = img_path.stem
        mask_path = RAW_DIR / "annotations" / f"{sample_id}.json"
        if not mask_path.exists():
            continue
        meta = process_sample(img_path, mask_path, sample_id)
        samples.append(meta)

    # Save metadata
    with open(PREP_DIR / "metadata.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Done! Preprocessed {len(samples)} samples.")

if __name__ == "__main__":
    main()