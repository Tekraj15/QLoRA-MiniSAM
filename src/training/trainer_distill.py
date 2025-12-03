# Training loop with teacher-student
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel, BitsAndBytesConfig
from src.distillation.loss import DistillationLoss
from src.utils.logger import init_wandb

class DistillationTrainer:
    def __init__(self, cfg, student_model, train_dataset):
        self.cfg = cfg
        self.device = cfg.train.device
        
        # 1. Setup Student (Dense, Trainable)
        self.student = student_model.to(self.device)
        self.student.train()

        # 2. Setup Teacher (Frozen)
        # Check if CUDA is available for 4-bit quantization
        if torch.cuda.is_available():
            print("CUDA detected. Loading Teacher (ViT-H) in 4-bit NF4...")
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.teacher = SamModel.from_pretrained(
                "facebook/sam-vit-huge",
                quantization_config=nf4_config,
                device_map=self.device
            )
        else:
            print(f"CUDA not detected (Device: {self.device}). Loading Teacher (ViT-H) in full precision...")
            # On MPS/CPU, we cannot use 4-bit quantization from bitsandbytes
            self.teacher = SamModel.from_pretrained(
                "facebook/sam-vit-huge"
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
                # Assuming dataset returns raw images/points
                inputs = self.processor(
                    batch["image"], 
                    input_points=batch["prompt"], 
                    return_tensors="pt"
                ).to(self.device)

                # --- Teacher Forward (Frozen) ---
                with torch.no_grad():
                    t_outputs = self.teacher(
                        pixel_values=inputs.pixel_values,
                        input_points=inputs.input_points,
                        multimask_output=False
                    )
                    # Explicitly get features if not in output (depends on HF version)
                    t_features = self.teacher.vision_encoder(inputs.pixel_values).last_hidden_state
                    
                    teacher_pack = {
                        'pred_masks': t_outputs.pred_masks,
                        'vision_features': t_features
                    }

                # --- Student Forward (Trainable) ---
                s_outputs = self.student.model(
                    pixel_values=inputs.pixel_values,
                    input_points=inputs.input_points,
                    multimask_output=False
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