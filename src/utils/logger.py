import wandb
import logging
from datetime import datetime
from omegaconf import OmegaConf

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

def init_wandb(cfg):
    wandb.init(
        project="lora-minisam",
        name=f"{cfg.train.mode}_{datetime.now().strftime('%H%M')}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )