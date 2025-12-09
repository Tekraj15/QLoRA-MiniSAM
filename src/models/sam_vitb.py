# Student model (ViT-B + SAM head)

import torch
import torch.nn as nn
from transformers import SamModel, SamConfig

class StudentSAM(nn.Module):
    def __init__(self, model_name="facebook/sam-vit-base"):
        super().__init__()
        print(f"Initializing Dense Student Model: {model_name}")
        
        # Load standard ViT-B from HuggingFace
        # This is NOT quantized yet. It is FP32/BF16 and fully trainable.
        self.model = SamModel.from_pretrained(model_name)
        
        # Enable gradients for all parameters (Full Fine-Tuning/Distillation)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values, input_points=None, input_boxes=None, input_labels=None):
        # Pass inputs to the HuggingFace model
        outputs = self.model(
            pixel_values=pixel_values,
            input_points=input_points,
            input_boxes=input_boxes,
            input_labels=input_labels,
            multimask_output=False # We usually want single mask for medical ground truth
        )
        
        # Return a dictionary for the Loss function
        return {
            'pred_masks': outputs.pred_masks,
            'iou_scores': outputs.iou_scores,
            'vision_features': outputs.vision_features # HF SamModel usually returns this only if configured
        }
    
    def get_vision_features(self, pixel_values):
        # Helper to extract features explicitly
        return self.model.vision_encoder(pixel_values).last_hidden_state

    def save_pretrained(self, save_directory):
        """Delegates saving to the underlying HuggingFace model."""
        self.model.save_pretrained(save_directory)