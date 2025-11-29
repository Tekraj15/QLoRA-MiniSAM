### QLoRA-MiniSAM: A Quantized LoRA-Adapted Lightweight SAM for Medical Image Segmentation

## Objective
Develop **QLoRA-MiniSAM**, a **lightweight, 4-bit quantized**, and **LoRA-adapted** version of the Segment Anything Model (SAM)  tailored for medical image segmentation (MIS) by combining knowledge distillation from the original SAM with **ViT-H** into a compact **ViT-B encoder** and Low-Rank Adaptation (LoRA) for modality-specific fine-tuning, achieving near-SOTA zero-shot and few-shot segmentation performance (**≥90% of ViT-H’s Dice score**) on COSMOS-1050K while reducing:
- **Inference latency**: **~7×**
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
    subgraph Input Processing
    I["Input Image<br/>(1024×1024)"] --> P["Patch Embeddings"]
    end

    subgraph QLoRA-MiniSAM Encoder
    direction TB
    W["Frozen Weights<br/>(4-bit NF4)"] --"De-quantize"--> M["MatMul (BF16)"]
    L["LoRA Adapters<br/>(Trainable Rank r=8)"] --"Scale & Add"--> M
    P --> M
    M --> F["Transformer Blocks<br/>(12 Layers)"]
    end

    subgraph Lightweight Decoding
    F --> E["Image Embeddings<br/>(64×64×256)"]
    
    U["User Prompts<br/>(Points/Boxes)"] --> PE["Prompt Encoder"]
    
    E --> D["Mask Decoder"]
    PE --> D
    D --> O["Segmentation Masks"]
    end

    style W fill:#ffeb3b,stroke:#fbc02d,stroke-width:2px
    style L fill:#ff7043,stroke:#d84315,stroke-width:2px
    style D fill:#66bb6a,stroke:#2e7d32,stroke-width:2px
```

# 2. Training Workflow:


```mermaid
flowchart LR
    subgraph Phase 1: Knowledge Distillation
    A["COSMOS Data"] --> B["Teacher: ViT-H<br/>(4-bit NF4 Frozen)"]
    A --> C["Student: ViT-B<br/>(BF16 Dense Trainable)"]
    B --"Embeddings"--> D["Loss Calculation"]
    C --"Embeddings"--> D
    D --> C
    end
    
    subgraph Phase 2: QLoRA Fine-Tuning
    C --> E["Distilled ViT-B<br/>(Frozen 4-bit NF4)"]
    E --> F["+ LoRA Adapters<br/>(Trainable)"]
    F --> G["Task Loss (Dice/Focal)"]
    end
```