# Dedicated adapter evaluation
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.models.qlora_minisam import QLoRAMiniSAM
from data.cosmo1050k.data_loader import COSMOSDataset
from utils.metrics import dice_score
from utils.viz import visualize_prediction
from torch.utils.data import DataLoader
import torch
import os

# Register config
cs = ConfigStore.instance()
cs.store(name="adapter_inference", node={
    "defaults": [
        {"data": "cosmo"},
        {"model": "qlora"},
        {"inference": "adapter"}
    ],
    "paths": {}, "dataset": {}, "inference": {}
})

@hydra.main(config_path="../../config", config_name="adapter_inference", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.train.device)
    os.makedirs(cfg.paths.outputs, exist_ok=True)

    print(f"Loading QLoRA adapter for modality: {cfg.inference.modality}")
    print(f"Adapter path: {cfg.inference.adapter_path}")

    # Load model with adapter CT/MRI/US
    model = QLoRAMiniSAM.load_adapter(
        cfg=cfg,
        path=cfg.inference.adapter_path,
        modality=cfg.inference.modality
    ).to(device)
    model.eval()

    # Load Dataset
    dataset = COSMOSDataset(cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    for i, batch in enumerate(loader):
        img = batch["image"].to(device)
        prompt = batch["prompt"]
        gt = batch["mask"].to(device)
        sample_id = batch["sample_id"][0]

        with torch.no_grad():
            pred = model(img, **prompt)[0]  # [1, H, W]

        # Computes Dice
        dice = dice_score(pred, gt).item()
        results.append({"sample_id": sample_id, "dice": dice})

        # Save visualization
        save_path = f"{cfg.paths.outputs}/vis/{sample_id}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        visualize_prediction(img[0], pred[0], gt[0], save_path=save_path)

        if i % 50 == 0:
            print(f"[{i}] {sample_id} | Dice: {dice:.3f}")

    # Final stats
    avg_dice = sum(r["dice"] for r in results) / len(results)
    print(f"\nFINAL → Modality: {cfg.inference.modality} | Avg Dice: {avg_dice:.3f}")

    # Save results
    with open(f"{cfg.paths.outputs}/adapter_results_{cfg.inference.modality}.txt", "w") as f:
        f.write(f"Modality: {cfg.inference.modality}\n")
        f.write(f"Average Dice: {avg_dice:.3f}\n")
        for r in results:
            f.write(f"{r['sample_id']}: {r['dice']:.3f}\n")

if __name__ == "__main__":
    main()