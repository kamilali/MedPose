#!/bin/bash
export PYTHONWARNINGS="ignore"

export NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$NGPUS $HOME/MedPose/tools/train_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_baseline.yaml SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 4 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 4000
