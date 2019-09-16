#!/bin/bash
export NGPUS=1
CUDA_VISIBLE_DEVICES=0 python $HOME/medpose_network/tools/test_net.py --config-file $HOME/medpose_network/configs/medpose_config_1x.yaml --ckpt /pasteur/u/kamil/models_medpose/model_0222500.pth TEST.IMS_PER_BATCH 1
