#!/bin/bash
PYTHONPATH=. python src/training/trainer_qlora.py \
  train=qlora \
  dataset.modality=MRI \
  dataset.split=train \
  train.epochs=20