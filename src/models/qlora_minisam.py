# QLoRA Injection Module for MiniSAM
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

class QLoRAMiniSAM(nn.Module):
    def __init__(self, cfg, modality="CT"):
        super().__init__()
        self.cfg = cfg
        self.modality = modality

        # 4-bit quantization
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.qlora.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.qlora.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.qlora.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.qlora.quantization.bnb_4bit_use_double_quant,
        )

        # Load ViT-B with 4-bit
        self.sam = sam_model_registry["vit_b"](
            checkpoint=cfg.sam.vit_b_checkpoint,
            quantization_config=quant_cfg
        )

        # Prepare for QLoRA
        self.sam = prepare_model_for_kbit_training(self.sam)

        # LoRA config
        lora_cfg = LoraConfig(
            r=cfg.qlora.r,
            lora_alpha=cfg.qlora.alpha,
            target_modules=cfg.qlora.target_modules,
            lora_dropout=cfg.qlora.dropout,
            bias="none",
            modules_to_save=["mask_decoder"]
        )

        self.sam = get_peft_model(self.sam, lora_cfg)
        self.sam.print_trainable_parameters()

        # Freeze everything except LoRA
        for name, param in self.sam.named_parameters():
            if not ("lora" in name or "modules_to_save" in name):
                param.requires_grad = False

    def forward(self, image, prompt):
        return self.sam(image, **prompt)

    def save_adapter(self, path):
        self.sam.save_pretrained(path)

    @classmethod
    def load_adapter(cls, cfg, path, modality):
        model = cls(cfg, modality)
        model.sam.load_adapter(path)
        return model