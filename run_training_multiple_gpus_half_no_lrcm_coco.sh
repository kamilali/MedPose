#!/bin/bash
export PYTHONWARNINGS="ignore"

export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS $HOME/MedPose/tools/train_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_encoder_no_lrcm_coco_pretrain.yaml SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 2 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
