"""
    Large-scale batch processing (e.g., 10K images). 
    Used for benchmarking, dataset generation.
"""
# src/inference/batch_seg.py
import hydra
from hydra.core.config_store import ConfigStore
from src.models.qlora_minisam import QLoRAMiniSAM
from data.cosmo1050k.data_loader import COSMOSDataset
from utils.metrics import dice_score
from torch.utils.data import DataLoader
import torch
import time
import os

cs = ConfigStore.instance()
cs.store(name="batch_seg", node={
    "defaults": ["data", "model", "inference"],
    "paths": {}, "dataset": {}, "inference": {}
})

@hydra.main(config_path="../../config", config_name="batch_seg", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.train.device)
    os.makedirs(cfg.paths.outputs, exist_ok=True)

    # Load adapter
    model = QLoRAMiniSAM.load_adapter(
        cfg=cfg,
        path=cfg.inference.adapter_path,
        modality=cfg.inference.modality
    ).to(device)
    model.eval()

    # Dataset
    dataset = COSMOSDataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg.inference.batch_size, shuffle=False)

    results = []
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(device)
            prompt = batch["prompt"]
            gt = batch["mask"].to(device)

            pred = model(img, **prompt)[0]
            dice = dice_score(pred, gt).item()
            results.append(dice)

            if i % 100 == 0:
                print(f"[{i}] Dice: {dice:.3f} | Avg: {sum(results)/len(results):.3f}")

    total_time = time.time() - start_time
    avg_dice = sum(results) / len(results)
    fps = len(results) / total_time

    print(f"\nBATCH RESULTS → {cfg.inference.modality.upper()}")
    print(f"Samples: {len(results)} | Avg Dice: {avg_dice:.3f} | FPS: {fps:.1f}")

    # Save
    with open(f"{cfg.paths.outputs}/batch_results.txt", "w") as f:
        f.write(f"Modality: {cfg.inference.modality}\n")
        f.write(f"Dice: {avg_dice:.3f}\n")
        f.write(f"FPS: {fps:.1f}\n")
        f.write(f"Time: {total_time:.1f}s\n")

if __name__ == "__main__":
    main()