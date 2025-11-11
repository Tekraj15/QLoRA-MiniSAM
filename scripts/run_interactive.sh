# Run Real-time interactive segmentation with specified modality (CT or MRI)
#!/bin/bash
MODALITY=$1
python src/inference/interactive_seg.py \
  inference.modality=$MODALITY \
  inference.adapter_path="checkpoints/qlora_${MODALITY}_final.pt"