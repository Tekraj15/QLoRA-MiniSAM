# Student model (ViT-B + SAM head)
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model

class MiniSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Load official ViT-B SAM (student)
        self.sam = sam_model_registry["vit_b"](
            checkpoint=cfg.sam.vit_b_checkpoint
        )
        # Freeze prompt encoder & mask decoder (distill only image encoder)
        for name, param in self.sam.named_parameters():
            if "prompt_encoder" in name or "mask_decoder" in name:
                param.requires_grad = False

    def forward(self, image, prompt):
        return self.sam(image, **prompt)

    def get_image_features(self, image):
        """Extract intermediate features for distillation"""
        with torch.no_grad():
            # Hook into ViT-B encoder
            features = self.sam.image_encoder(image)
        return features