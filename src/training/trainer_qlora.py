# QLoRA Training
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamProcessor
from src.models.qlora_minisam import QLoRAMiniSAM
from data.cosmo1050k.data_loader import COSMOSDataset 
import wandb

# --- Robust Medical Segmentation Loss ---
class MedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_masks, gt_mask):
        # pred_masks: [B, 1, 256, 256] (Logits)
        # gt_mask: [B, 1, 256, 256] (0 or 1)
        
        # 1. Focal/BCE Loss
        loss_bce = self.bce(pred_masks, gt_mask)
        
        # 2. Dice Loss
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_mask).sum()
        union = pred_probs.sum() + gt_mask.sum()
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        loss_dice = 1 - dice
        
        # Weighted sum
        return 0.5 * loss_bce + 0.5 * loss_dice, dice

class QLoRATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.train.device

        # 1. Dataset & Processor
        # We need the processor to handle resizing to 1024x1024
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        dataset = COSMOSDataset(cfg) # Ensure this yields raw images
        self.loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=self.collate_fn)

        # 2. Model (Loads Distilled Weights + LoRA)
        self.model = QLoRAMiniSAM(cfg).to(self.device)
        
        # 3. Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(cfg.train.lr)
        )
        
        self.criterion = MedSegLoss()

        if cfg.use_wandb:
            wandb.init(project="qlora-minisam", name=f"{cfg.dataset.modality}_adapter")

    def train(self):
        self.model.train()
        print(f"Starting QLoRA Fine-Tuning for {self.cfg.dataset.modality}...")
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            epoch_loss = 0.0
            epoch_dice = 0.0
            
            pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                # Process inputs
                inputs = self.processor(
                    batch["image"], 
                    input_points=batch["prompt"], 
                    return_tensors="pt"
                ).to(self.device)
                
                # GT Mask needs resizing to match SAM output (usually 256x256)
                # SAM outputs low-res masks (256x256) by default for efficiency
                gt_mask = batch["mask"].to(self.device).float()
                if gt_mask.shape[-1] != 256:
                     gt_mask = F.interpolate(gt_mask, size=(256, 256), mode='nearest')

                # Forward
                outputs = self.model(
                    pixel_values=inputs.pixel_values,
                    input_points=inputs.input_points,
                    multimask_output=False
                )
                
                # Output shape: [B, 1, 256, 256]
                pred_masks = outputs.pred_masks[:, 0, :, :] 

                # Loss
                loss, dice = self.criterion(pred_masks, gt_mask)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                pbar.set_postfix({"Loss": loss.item(), "Dice": dice.item()})
                
                if self.cfg.use_wandb:
                    wandb.log({"loss": loss.item(), "dice": dice.item()})

            avg_loss = epoch_loss/len(self.loader)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Avg Dice: {epoch_dice/len(self.loader):.4f}")
            
            # Save Adapter
            self.model.save_adapter(f"{self.cfg.paths.checkpoints}/qlora_{self.cfg.dataset.modality}_epoch{epoch}")

    def collate_fn(self, batch):
        return {
            "image": [item["image"] for item in batch],
            "prompt": [item["prompt"] for item in batch],
            "mask": torch.stack([item["mask"] for item in batch])
        }