#!/usr/bin/env bash
python deeplab/export_model.py \
--checkpoint_path=../../exp/xception2/train/model.ckpt-658 \
--export_path=../../exp/xception2/frozen_inference_graph.pb \
--num_classes=78 \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
