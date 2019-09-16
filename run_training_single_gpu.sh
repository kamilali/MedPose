#!/bin/bash
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python $HOME/medpose_network/tools/train_net.py --config-file $HOME/medpose_network/configs/medpose_config_1x.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
