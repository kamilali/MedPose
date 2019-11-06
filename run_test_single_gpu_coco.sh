#!/bin/bash
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python $HOME/MedPose/tools/test_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_coco.yaml --ckpt $SCRATCH/medpose-model-data/models_medpose_w_coco_single_frames_retry/model_0360000.pth TEST.IMS_PER_BATCH 1
