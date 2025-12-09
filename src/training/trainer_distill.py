# Training loop with teacher-student
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, BitsAndBytesConfig
from src.distillation.loss import DistillationLoss
from src.utils.logger import init_wandb
import hydra
from src.models.sam_vitb import StudentSAM
from data.cosmos1050k.data_loader import COSMOSDataset

class DistillationTrainer:
    def __init__(self, cfg, student_model, train_dataset):
        self.cfg = cfg
        self.device = cfg.train.device
        
        # 1. Setup Student (Dense, Trainable)
        self.student = student_model.to(self.device)
        self.student.train()

        # 2. Setup Teacher (Frozen)
        # Check if CUDA is available
        if torch.cuda.is_available():
            print("CUDA detected. Loading Teacher (ViT-B) in float16 (Windows compatibility)...")
            # Disable 4-bit for now due to bitsandbytes issues on Windows
            # Use float16 to save memory (ViT-B is smallest)
            self.teacher = SamModel.from_pretrained(
                "facebook/sam-vit-base",
                device_map=self.device,
                torch_dtype=torch.float16
            )
        else:
            print(f"CUDA not detected (Device: {self.device}). Loading Teacher (ViT-B) in full precision...")
            # On MPS/CPU, we cannot use 4-bit quantization from bitsandbytes
            self.teacher = SamModel.from_pretrained(
                "facebook/sam-vit-base"
            ).to(self.device)
        
        self.teacher.eval() # Freeze teacher

        # 3. Processor (Handles resizing to 1024x1024 and Norm)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # 4. Optimization
        self.criterion = DistillationLoss(
            alpha=cfg.train.feature_loss_weight,
            temperature=cfg.train.temperature
        )
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay)
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn # Use custom collator if needed
        )
        
        init_wandb(cfg)

    def train(self):
        print("Starting Distillation Phase...")
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                # Prepare inputs using Processor (handles batching of lists)
                # Parse prompts from dataset dicts
                prompts = batch["prompt"]
                input_boxes = None
                input_points = None
                
                if prompts[0]["type"] == "box":
                    # SAM expects list of list of list for boxes: (batch, num_boxes, 4)
                    # We have 1 box per image, so we wrap it in a list
                    input_boxes = [[p["box"]] for p in prompts]
                elif prompts[0]["type"] == "point":
                    # SAM expects list of list of list for points: (batch, num_points, 2)
                    input_points = [[[p["coord"]]] for p in prompts]

                inputs = self.processor(
                    batch["image"], 
                    input_points=input_points,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                ).to(self.device)

                # --- Teacher Forward (Frozen) ---
                with torch.no_grad():
                    # Cast inputs to teacher's dtype (float16) for the teacher forward pass
                    teacher_inputs = {k: v.to(dtype=self.teacher.dtype) if torch.is_floating_point(v) else v 
                                      for k, v in inputs.items()}
                    
                    t_outputs = self.teacher(
                        multimask_output=False,
                        **teacher_inputs
                    )
                    # Explicitly get features if not in output (depends on HF version)
                    t_features = self.teacher.vision_encoder(teacher_inputs["pixel_values"]).last_hidden_state
                    
                    teacher_pack = {
                        'pred_masks': t_outputs.pred_masks,
                        'vision_features': t_features
                    }

                # --- Student Forward (Trainable) ---
                # Student is float32 (default), so use original inputs
                s_outputs = self.student.model(
                    multimask_output=False,
                    **inputs
                )
                s_features = self.student.model.vision_encoder(inputs.pixel_values).last_hidden_state
                
                student_pack = {
                    'pred_masks': s_outputs.pred_masks,
                    'vision_features': s_features
                }

                # --- Distillation Step ---
                loss = self.criterion(student_pack, teacher_pack)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                
                if self.cfg.use_wandb:
                    wandb.log({"train_loss": loss.item()})

            # Save Checkpoint (HF Format for Phase 2)
            save_path = f"{self.cfg.paths.checkpoints}/distilled_student_epoch{epoch}"
            self.student.save_pretrained(save_path)
            self.processor.save_pretrained(save_path) # Save processor too for completeness
            print(f"Epoch {epoch} complete. Saved to {save_path}. Loss: {epoch_loss / len(self.train_loader):.4f}")

    def collate_fn(self, batch):
        # Simple helper to keep list structure for the Processor
        return {
            "image": [item["image"] for item in batch],
            "prompt": [item["prompt"] for item in batch], # Points
            "mask": [item["mask"] for item in batch]
        }

@hydra.main(config_path="../../config", config_name="default", version_base="1.3")
def main(cfg):
    # 1. Dataset
    dataset = COSMOSDataset(cfg)
    
    # 2. Student Model
    student = StudentSAM()
    
    # 3. Trainer
    trainer = DistillationTrainer(cfg, student, dataset)
    
    # 4. Train
    trainer.train()

if __name__ == "__main__":
    main()