#!/bin/bash
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python $HOME/MedPose/tools/test_net.py --config-file $HOME/MedPose/configs/medpose_config_1x.yaml --ckpt $SCRATCH/medpose-model-data/models_medpose_w_posetrack_retry/model_0007500.pth TEST.IMS_PER_BATCH 1
