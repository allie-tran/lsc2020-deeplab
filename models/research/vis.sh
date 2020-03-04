#!/usr/bin/env bash
export DATE=2016-08-15
cd deeplab/datasets && python build_ade20k_data.py ${DATE} && cd ../../ && \
python deeplab/vis.py \
--folder=${DATE} \
--logtostderr \
--vis_split="val" \
--model_variant="xception_65" \
--min_resize_value=513 \
--max_resize_value=513 \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--resize_factor=16 \
--decoder_output_stride=4 \
--dataset="ade20k" \
--colormap_type='ade20k' \
--checkpoint_dir=../../exp/xception/train \
--vis_logdir=../../exp/xception2/vis \
--dataset_dir=../../vis/tf/${DATE} \
--max_number_of_iterations=1 \
--also_save_raw_predictions && \
rm -rf ../../vis/tf/${DATE} && \
rm -rf ../../vis/${DATE}/ && \
mkdir ../../vis/${DATE}/ && \
mv ../../exp/xception2/vis/raw_segmentation_results/* ../../vis/${DATE}/ && \
python seg2feature.py ${DATE} && \
rm -rf ../../vis/${DATE}/
echo $DATE >> done.txt
