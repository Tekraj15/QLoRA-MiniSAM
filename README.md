### LoRA-MiniSAM - A lightweight, parameter-efficient adaptation of the Segment Anything Model (SAM)

## Objective
The primary objective of LoRA-MiniSAM is to develop a lightweight, parameter-efficient adaptation of the Segment Anything Model (SAM) tailored for medical image segmentation (MIS) by combining knowledge distillation from the original SAM with ViT-H into a compact ViT-B-based encoder and Low-Rank Adaptation (LoRA) for modality-specific fine-tuning. We aim to achieve near-SOTA zero- and few-shot segmentation performance (≥80% of full SAM’s Dice score) while reducing inference latency by 10–20×, memory footprint by >90%, and training cost by >95% compared to full fine-tuning. The system will enable real-time, on-device deployment in clinical environments and scalable multi-modal adaptation across diverse imaging modalities (CT, MRI, ultrasound, etc.), addressing the critical barriers of computational overhead and domain shift identified in large-scale evaluations of SAM on medical data.

## Introduction
The Segment Anything Model (SAM) [Kirillov et al., 2023] represents a landmark in foundation models for vision, demonstrating unprecedented zero-shot generalization to arbitrary object segmentation via prompt-based interaction. Trained on over 11 million images and 1 billion masks, SAM leverages a Vision Transformer (ViT) backbone—particularly the high-capacity ViT-H/14 variant—to extract rich, hierarchical visual representations, enabling robust performance across diverse natural image domains. Recent large-scale evaluations on medical imaging [Huang et al., 2024], however, reveal significant limitations when applying SAM directly to medical image segmentation (MIS): while SAM exhibits promising results on well-defined anatomical structures, it struggles with amorphous lesions, fine boundaries, low-contrast regions, and modality shifts. Moreover, the computational demands of ViT-H (∼190 GFLOPs, 632M parameters) render it impractical for clinical deployment, especially on resource-constrained devices or in real-time workflows.


To bridge this gap, parameter-efficient fine-tuning (PEFT) strategies such as Low-Rank Adaptation (LoRA) [Hu et al., 2021] have emerged as powerful tools for adapting large foundation models with minimal trainable parameters. Concurrently, knowledge distillation into lightweight architectures (e.g., MobileViT, TinyViT) has shown success in compressing SAM for edge deployment. Yet, no prior work has systematically combined distilled lightweight encoders with LoRA-based modality adaptation to create a unified, efficient, and generalizable framework for MIS.


We introduce LoRA-MiniSAM, a novel hybrid architecture that:

- Distills the representational power of SAM’s ViT-H into a compact ViT-B/16 encoder (86M → reduced latent dimension),
Freezes the distilled backbone and injects LoRA modules into attention layers for modality-specific adaptation,
Preserves SAM’s prompt encoder and mask decoder for interactive zero-shot capability.

- This design enables plug-and-play adapters per imaging modality, supports few-shot learning with <100 labeled samples, and achieves real-time inference (>30 FPS on consumer GPUs). Being evaluated on the COSMOS 1050K benchmark [Huang et al., 2024], LoRA-MiniSAM targets 4–6% Dice improvement over zero-shot SAM with <5M trainable parameters and <10 GFLOPs per forward pass, making it a practical foundation model for clinical MIS.