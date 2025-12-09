# Feature + Mask distillation loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.mse = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs):
        """
        HuggingFace SamModel outputs return:
        - iou_scores
        - pred_masks
        - vision_embedding (access this manually if not returned by default)
        """
        
        # 1. Feature Distillation (Image Encoder Embeddings)
        # Assume the trainer extracts the image embeddings (B, 256, 64, 64)
        s_feat = student_outputs['vision_features']
        # Cast Teacher features to Student's dtype (float32) to avoid Half/Float mismatch in backward
        t_feat = teacher_outputs['vision_features'].to(dtype=s_feat.dtype)
        
        # Normalize features for stability (Cosine Similarity via MSE)
        s_feat_norm = F.normalize(s_feat, dim=1)
        t_feat_norm = F.normalize(t_feat, dim=1)
        feat_loss = self.mse(s_feat_norm, t_feat_norm)

        # 2. Mask Distillation (Logits)
        # Using the first mask (index 0) which is usually the most confident for unambiguous prompts
        s_masks = student_outputs['pred_masks'][:, 0, :, :] # (B, H, W)
        # Cast Teacher masks to Student's dtype
        t_masks = teacher_outputs['pred_masks'][:, 0, :, :].to(dtype=s_masks.dtype)

        # Flatten for KL Div
        # s_masks is (B, 1, H, W) -> squeeze channel dim if present
        if len(s_masks.shape) == 4:
             s_masks = s_masks.squeeze(1)
             t_masks = t_masks.squeeze(1)

        B, H, W = s_masks.shape
        s_logits = s_masks.view(B, -1) / self.T
        t_logits = t_masks.view(B, -1) / self.T

        mask_loss = self.kld(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1)
        ) * (self.T ** 2)

        return self.alpha * feat_loss + (1 - self.alpha) * mask_loss