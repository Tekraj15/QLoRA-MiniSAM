#!/bin/bash
PYTHONPATH=. python src/training/trainer_qlora.py \
  train=qlora \
  dataset.modality=CT \
  dataset.split=train \
  train.epochs=20 \
  train.batch_size=8