#!/bin/bash
export NGPUS=1
export PYTHONPATH=$HOME/MedPose/maskrcnn_benchmark/data/datasets/evaluation/poseval/py-motmetrics/:$PYTHONPATH

# save files for testing
CUDA_VISIBLE_DEVICES=0 python $HOME/MedPose/tools/test_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_encoder.yaml --ckpt $SCRATCH/medpose-model-data/models_medpose_half_res101_w_posetrack_only/model_0006000.pth TEST.IMS_PER_BATCH 1

