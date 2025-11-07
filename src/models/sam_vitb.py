# Student model (ViT-B + SAM head)
# src/models/sam_vitb.py
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

class QLoRAMiniSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 4-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load ViT-B with 4-bit
        self.sam = sam_model_registry["vit_b"](
            checkpoint=cfg.sam.vit_b_checkpoint,
            quantization_config=quantization_config
        )

        # Prepare for QLoRA
        self.sam = prepare_model_for_kbit_training(self.sam)

        # LoRA config
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout,
            bias="none",
            modules_to_save=["mask_decoder"]  # keep decoder in FP16
        )

        # Apply LoRA
        self.sam = get_peft_model(self.sam, lora_config)

        # Freeze prompt encoder & mask decoder (except saved)
        for name, param in self.sam.named_parameters():
            if "prompt_encoder" in name or ("mask_decoder" in name and "modules_to_save" not in name):
                param.requires_grad = False

    def forward(self, image, prompt):
        return self.sam(image, **prompt)

    def get_image_features(self, image):
        return self.sam.image_encoder(image)