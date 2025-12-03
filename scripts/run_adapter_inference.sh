MODALITY=$1
EPOCH=${2:-final}

PYTHONPATH=. python src/inference/adapter_inference.py \
  inference.modality=$MODALITY \
  inference.adapter_path="checkpoints/qlora_${MODALITY}_epoch${EPOCH}.pt" \
  dataset.split=valid \
  dataset.modality=$MODALITY \
  paths.outputs="outputs/adapter_inference"