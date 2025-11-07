# QLoRA Training
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.qlora_minisam import QLoRAMiniSAM
from data.cosmo1050k.data_loader import COSMOSDataset
from utils.metrics import dice_score
import wandb

class QLoRATrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)

        # Dataset
        dataset = COSMOSDataset(cfg)
        self.loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

        # Model
        self.model = QLoRAMiniSAM(cfg, modality=cfg.dataset.modality).to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.train.lr
        )

        wandb.init(project="qlora-minisam", name=f"{cfg.dataset.modality}_adapter")

    def train(self):
        self.model.train()
        for epoch in range(1, cfg.train.epochs + 1):
            epoch_loss = 0.0
            for batch in tqdm(self.loader, desc=f"Epoch {epoch}"):
                img = batch["image"].to(self.device)
                prompt = batch["prompt"]
                gt = batch["mask"].to(self.device)

                # Forward
                pred = self.model(img, **prompt)[0]
                loss = 1 - dice_score(pred, gt)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                wandb.log({"loss": loss.item(), "dice": 1 - loss.item()})

            print(f"Epoch {epoch} | Loss: {epoch_loss/len(self.loader):.4f}")
            self.model.save_adapter(f"{cfg.paths.checkpoints}/qlora_{cfg.dataset.modality}_epoch{epoch}.pt")