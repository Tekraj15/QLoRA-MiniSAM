#!/bin/bash
MODALITY=$1
BATCH_SIZE=${2:-8}

python src/inference/batch_seg.py \
  inference.modality=$MODALITY \
  inference.adapter_path="checkpoints/qlora_${MODALITY}_final.pt" \
  inference.batch_size=$BATCH_SIZE \
  dataset.split=test \
  paths.outputs="outputs/batch_seg"