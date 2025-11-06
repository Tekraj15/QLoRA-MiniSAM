"""
zero-shot performance benchmark for the original SAM models (ViT-H and ViT-B) on the COSMOS-1050K dataset.
Used to reproduce the ooriginal SAM paper’s zero-shot numbers before claiming any gains.
"""
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from segment_anything import sam_model_registry
import torch
from torch.utils.data import DataLoader
from data.cosmo1050k.data_loader import COSMOSDataset
from utils.metrics import dice_score
from tqdm import tqdm
import os

# Register config
cs = ConfigStore.instance()
cs.store(name="config", node={
    "defaults": [
        {"data": "cosmo"},
        {"model": "sam"},
        {"eval": "baseline"}
    ],
    "project": {}, "paths": {}, "dataset": {}, "sam": {}, "eval": {}
})

@hydra.main(config_path="../../config", config_name="default", version_base="1.3")
def main(cfg):
    torch.manual_seed(cfg.project.seed)
    device = torch.device(cfg.sam.device)

    # Load models
    sam_h = sam_model_registry["vit_h"](checkpoint=cfg.sam.vit_h_checkpoint).to(device).eval()
    sam_b = sam_model_registry["vit_b"](checkpoint=cfg.sam.vit_b_checkpoint).to(device).eval()

    # Dataset
    dataset = COSMOSDataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    results = []
    for i, batch in enumerate(tqdm(loader, desc="Zero-shot Eval")):
        img = batch["image"].to(device)
        prompt = batch["prompt"]
        gt = batch["mask"].to(device)

        with torch.no_grad():
            # SAM forward expects dict with 'type' and 'box' or 'point'
            if prompt["type"] == "box":
                input_prompt = {"input_boxes": torch.tensor([prompt["box"]]).to(device)}
            elif prompt["type"] == "point":
                input_prompt = {
                    "input_points": torch.tensor([[prompt["coord"]]]).to(device),
                    "input_labels": torch.tensor([prompt["label"]]).to(device)
                }
            else:
                input_prompt = {}

            pred_h = sam_h(img, **input_prompt)[0]  # [1, H, W]
            pred_b = sam_b(img, **input_prompt)[0]

        results.append({
            "dice_h": dice_score(pred_h, gt).item(),
            "dice_b": dice_score(pred_b, gt).item(),
        })

        if i % cfg.eval.log_interval == 0 and i > 0:
            print(f"Step {i}: H={results[-1]['dice_h']:.3f}, B={results[-1]['dice_b']:.3f}")

    # Final stats
    dice_h = sum(r["dice_h"] for r in results) / len(results)
    dice_b = sum(r["dice_b"] for r in results) / len(results)
    print(f"\nFINAL → ViT-H Dice: {dice_h:.3f}")
    print(f"       ViT-B Dice: {dice_b:.3f}")
    print(f"       Gap: {dice_h - dice_b:.3f}")

    # Save results
    os.makedirs(cfg.paths.outputs, exist_ok=True)
    with open(f"{cfg.paths.outputs}/baseline_results.txt", "w") as f:
        f.write(f"ViT-H: {dice_h:.3f}\nViT-B: {dice_b:.3f}\n")

if __name__ == "__main__":
    main()