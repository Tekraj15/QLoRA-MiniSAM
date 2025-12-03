#!/bin/bash
# scripts/train_distill.sh
PYTHONPATH=. python src/training/trainer_distill.py \
  train=distill \
  dataset.split=train \
  dataset.modality=CT \
  train.epochs=50 \
  train.batch_size=8 \
  train.device=cuda