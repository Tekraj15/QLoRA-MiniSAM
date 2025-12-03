# QLoRA Injection Module for MiniSAM
import torch
import torch.nn as nn
from transformers import SamModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class QLoRAMiniSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. Define 4-bit NF4 Config (The "Q" in QLoRA)
        bnb_config = None
        if torch.cuda.is_available():
            print("CUDA detected. Using 4-bit NF4 quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            print(f"CUDA not detected (Device: {cfg.train.device}). Using full precision (LoRA only)...")

        print(f"Loading Distilled Student from: {cfg.paths.distilled_checkpoint}")
        
        # 2. Load the DISTILLED Student; assuming it was saved using model.save_pretrained()
        
        self.model = SamModel.from_pretrained(
            cfg.paths.distilled_checkpoint, # Path to your local best_student folder
            quantization_config=bnb_config,
            device_map=cfg.train.device if torch.cuda.is_available() else None # MPS doesn't like device_map="auto" sometimes
        )
        
        if not torch.cuda.is_available():
             self.model = self.model.to(cfg.train.device)

        # 3. Prepare for k-bit training (freezes layers, casts layernorms to fp32)
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # If not quantizing, we still want to freeze base model for LoRA
            # But get_peft_model will handle some of this. 
            # We explicitly freeze here to be safe and match QLoRA behavior
            for param in self.model.parameters():
                param.requires_grad = False

        # 4. LoRA Config
        # "q_proj", "v_proj" - HF SAM Attention layers
        lora_config = LoraConfig(
            r=cfg.qlora.r,
            lora_alpha=cfg.qlora.alpha,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=cfg.qlora.dropout,
            bias="none",
            modules_to_save=["mask_decoder"] # fully fine-tune the decoder
        )

        # 5. Inject Adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, pixel_values, input_points=None, input_boxes=None, multimask_output=False):
        # Pass directly to HF model
        return self.model(
            pixel_values=pixel_values, 
            input_points=input_points,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )

    def save_adapter(self, path):
        # Saves only the LoRA weights (few MBs)
        self.model.save_pretrained(path)