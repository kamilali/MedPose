#!/bin/bash
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python $HOME/MedPose/tools/test_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_baseline_coco_posttrain.yaml --ckpt $SCRATCH/medpose-model-data/models_kprcnn_baseline_res101_coco_posttrained/model_final.pth TEST.IMS_PER_BATCH 1
