#!/bin/bash
PYTHONPATH=. python src/inference/baseline_eval.py \
  data=cosmo \
  dataset.modality=CT \
  eval.num_samples=100