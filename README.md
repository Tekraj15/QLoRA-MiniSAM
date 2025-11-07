### QLoRA-MiniSAM: A Quantized LoRA-Adapted Lightweight SAM for Medical Image Segmentation

## Objective
Develop **QLoRA-MiniSAM**, a **lightweight, 4-bit quantized**, and **LoRA-adapted** version of the Segment Anything Model (SAM)  tailored for medical image segmentation (MIS) by combining knowledge distillation from the original SAM with **ViT-H** into a compact **ViT-B encoder** and Low-Rank Adaptation (LoRA) for modality-specific fine-tuning, achieving near-SOTA zero-shot and few-shot segmentation performance (**≥90% of ViT-H’s Dice score**) on COSMOS-1050K while reducing:
- **Inference latency**: **10–20×**
- **Memory footprint**: **>90%**
- **Training cost**: **>95%**

The system will enable **real-time, on-device MIS** and **plug-and-play modality adapters** (CT, MRI, US) with **<1GB VRAM**.

## Introduction
The Segment Anything Model (SAM) [Kirillov et al., 2023] represents a landmark in foundation models for vision, demonstrating unprecedented zero-shot generalization to arbitrary object segmentation via prompt-based interaction. Trained on over 11 million images and 1 billion masks, SAM leverages a Vision Transformer (ViT) backbone—particularly the high-capacity ViT-H/14 variant—to extract rich, hierarchical visual representations, enabling robust performance across diverse natural image domains. Recent large-scale evaluations on medical imaging [Huang et al., 2024], however, reveal significant limitations when applying SAM directly to medical image segmentation (MIS): while SAM exhibits promising results on well-defined anatomical structures, it struggles with amorphous lesions, fine boundaries, low-contrast regions, and modality shifts. Moreover, the computational demands of ViT-H (∼190 GFLOPs, 632M parameters) render it impractical for clinical deployment, especially on resource-constrained devices or in real-time workflows. 

Concurrently, knowledge distillation into lightweight architectures (e.g., MobileViT, TinyViT) has shown success in compressing SAM for edge deployment. Yet, no prior work has systematically combined distilled lightweight encoders with LoRA-based modality adaptation to create a unified, efficient, and generalizable framework for MIS.


To bridge this gap, **QLoRA-MiniSAM** combines:
1. **Knowledge distillation** from ViT-H → **ViT-B** (Mini-SAM)
2. **4-bit quantization** (NF4 + double quant) via **QLoRA**
3. **LoRA adapters** for **modality-specific fine-tuning**


This design yields a **deployable foundation model** for MIS i.e., it enables

- plug-and-play adapters per imaging modality, 

- supports few-shot learning with <100 labeled samples, 

- and achieves real-time inference (>30 FPS on consumer GPUs). 


Being Evaluated on the COSMOS-1050K benchmark [Huang et al., 2024], QLoRA-MiniSAM targets 4–6% Dice improvement over zero-shot SAM with <5M trainable parameters, <10 GFLOPs per forward pass, and >70% reduction in memory via 4-bit quantization, making it a practical, deployable foundation model for clinical MIS.


## Detailed Methodology
# 1. Architecture Overview

```mermaid
flowchart TD
    A[Input Image\n(1024×1024)] --> B[ViT-B Encoder\n(4-bit Quantized)]
    B --> C[Image Features\n(256×64×64)]
    
    D[Prompt\n(Box/Point)] --> E[Prompt Encoder]
    E --> F[Prompt Tokens]
    
    C --> G[Mask Decoder\n(2-layer Transformer)]
    F --> G
    G --> H[Mask Logits\n(3×1024×1024)]
    H --> I[Final Mask\n(Sigmoid + Argmax)]

    style B fill:#ffeb3b,stroke:#f57c00
    style G fill:#4caf50,stroke:#2e7d32
```

# 2. Training Workflow:
```mermaid
flowchart LR
    A[COSMOS Train Split] --> B[DataLoader]
    B --> C[Teacher: ViT-H\n(Frozen)]
    B --> D[Student: ViT-B\n(4-bit + LoRA)]
    C --> E[Teacher Features + Mask]
    D --> F[Student Features + Mask]
    E & F --> G[Distillation Loss\n(Feat + Mask KL)]
    G --> H[AdamW\n(lr=1e-4)]
    H --> D
```