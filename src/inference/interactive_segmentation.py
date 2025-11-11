"""
    Interactive segmentation demo (e.g., click on image → real-time mask). 
    Used for demos, videos, clinician feedback.
"""
import torch
from src.models.qlora_minisam import QLoRAMiniSAM
from utils.viz import visualize_prediction
import hydra

@hydra.main(config_path="../../config", config_name="default")
def main(cfg):
    device = torch.device(cfg.train.device)

    # Load adapter (CT, MRI, US)
    model = QLoRAMiniSAM.load_adapter(
        cfg=cfg,
        path=f"{cfg.paths.checkpoints}/qlora_{cfg.inference.modality}_final.pt",
        modality=cfg.inference.modality
    ).to(device)
    model.eval()

    # Example: single image
    img = torch.randn(1, 3, 1024, 1024).to(device)  # replace with real input
    prompt = {"input_boxes": torch.tensor([[[100, 100, 500, 500]]]).to(device)}

    with torch.no_grad():
        masks = model(img, **prompt)[0]
    
    visualize_prediction(img[0], masks[0], title=f"QLoRA-MiniSAM [{cfg.inference.modality}]")