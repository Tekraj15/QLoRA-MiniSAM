# Dedicated adapter evaluation
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from src.models.qlora_minisam import QLoRAMiniSAM
from data.cosmos1050k.data_loader import COSMOSDataset
from src.utils.metrics import dice_score
from src.utils.viz import visualize_prediction
from torch.utils.data import DataLoader
import torch
import os

# Register config
cs = ConfigStore.instance()
cs.store(name="adapter_inference", node={
    "defaults": [
        "_self_",
        {"data": "cosmo"},
        {"model": "qlora"},
        {"inference": "adapter"}
    ],
    "paths": {
        "outputs": "outputs/inference",
        "distilled_checkpoint": "./checkpoints/distilled_student_epoch1", # Required for loading base model
        "data_root": "./data/cosmos1050k"
    }, 
    "dataset": {"split": "valid", "modality": "all", "size": 256, "prompt_type": "box"}, 
    "train": {"device": "cuda"}, # Add train config for device access
    "inference": {
        "modality": "???",
        "adapter_path": "???"
    }
})

@hydra.main(config_path="../../config", config_name="adapter_inference", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.train.device)
    os.makedirs(cfg.paths.outputs, exist_ok=True)

    print(f"Adapter path: {cfg.inference.adapter_path}")

    # Use processor for resizing
    from transformers import SamProcessor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

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
        # img is [1, 1, 256, 256] or similar (dataset output)
        # processor expects list of numpy arrays or PIL images
        
        # We need to unbatch because processor handles batching
        img_np = batch["image"][0].numpy().transpose(1, 2, 0) # [H, W, C]
        
        prompt = batch["prompt"]
        gt = batch["mask"].to(device)
        sample_id = batch["sample_id"][0]

        input_boxes = None
        input_points = None
        
        # Parse prompts
        p_type = prompt["type"][0] # Extract string from list
        
        if p_type == "box":
             box_data = prompt["box"]
             # Check if it is a list of tensors (columns) from default_collate
             if isinstance(box_data, list):
                 # Stack columns: [tensor([x1]), tensor([y1])...] -> tensor([[x1, y1, x2, y2]])
                 box_tensor = torch.stack(box_data, dim=1)
             elif isinstance(box_data, torch.Tensor):
                 box_tensor = box_data
             else:
                 # Fallback
                 box_tensor = torch.tensor(box_data)
             
             # Convert to list for processor: [[x1, y1, x2, y2]] (batch size 1)
             # Processor expects list of list of list for batch? 
             # For batch size 1: [[[x1, y1, x2, y2]]]
             input_boxes = [box_tensor[0].tolist()] # box_tensor[0] is [x1, y1, x2, y2]
             # Wrap in another list for "one box per image" logic if processor expects it
             input_boxes = [input_boxes] # -> [[[x1, y1, x2, y2]]]
             
        elif p_type == "point":
             coord_data = prompt["coord"]
             if isinstance(coord_data, list):
                 coord_tensor = torch.stack(coord_data, dim=1) # [1, 2]
             elif isinstance(coord_data, torch.Tensor):
                 coord_tensor = coord_data
                 
             input_points = [[[coord_tensor[0].tolist()]]]

        # Process inputs
        inputs = processor(
            img_np,
            input_boxes=input_boxes,
            input_points=input_points,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(
                pixel_values=inputs.pixel_values.to(dtype=model.model.dtype),
                input_points=inputs.get("input_points"),
                input_boxes=inputs.get("input_boxes"),
                multimask_output=False
            )
            
            # The model wrapper returns raw HF outputs
            pred_logits = outputs.pred_masks[:, 0, :, :] # [1, 256, 256]
            
            pred = torch.sigmoid(pred_logits)
            pred = (pred > 0.5).float() # Binary mask

        # Computes Dice
        # gt shape: [1, 256, 256] -> we need [1, 256, 256]
        # In trainer we unsqueezed gt.
        if gt.dim() == 3:
            # gt is [1, 256, 256]
            pass 
        
        dice = dice_score(pred, gt).item()
        results.append({"sample_id": sample_id, "dice": dice})

        # Save visualization
        save_path = f"{cfg.paths.outputs}/vis/{sample_id}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Using batch["image"][0] which is [3, H, W]
        visualize_prediction(batch["image"][0], pred[0], gt[0], save_path=save_path)

        if i % 50 == 0:
            print(f"[{i}] {sample_id} | Dice: {dice:.3f}")

    # Final stats
    avg_dice = sum(r["dice"] for r in results) / len(results)
    print(f"\nFINAL -> Modality: {cfg.inference.modality} | Avg Dice: {avg_dice:.3f}")

    # Save results
    with open(f"{cfg.paths.outputs}/adapter_results_{cfg.inference.modality}.txt", "w") as f:
        f.write(f"Modality: {cfg.inference.modality}\n")
        f.write(f"Average Dice: {avg_dice:.3f}\n")
        for r in results:
            f.write(f"{r['sample_id']}: {r['dice']:.3f}\n")

if __name__ == "__main__":
    main()