#!/usr/bin/env bash
python deeplab/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--min_resize_value=513 \
--max_resize_value=513 \
--resize_factor=16 \
--decoder_output_stride=4 \
--dataset="ade20k" \
--checkpoint_dir=../../exp/xception2/train \
--eval_logdir=../../exp/xception2/test \
--dataset_dir=deeplab/datasets/ADE20K/tfrecord \
--max_number_of_evaluations=1