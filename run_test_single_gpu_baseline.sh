#!/bin/bash
export NGPUS=1
export PYTHONPATH=$HOME/MedPose/maskrcnn_benchmark/data/datasets/evaluation/poseval/py-motmetrics/:$PYTHONPATH

# save files for testing
CUDA_VISIBLE_DEVICES=0 python $HOME/MedPose/tools/test_net.py --config-file $HOME/MedPose/configs/medpose_config_1x_baseline.yaml --ckpt $SCRATCH/medpose-model-data/models_kprcnn_baseline_res101_w_posetrack_only/model_0006000.pth TEST.IMS_PER_BATCH 1

# evaluate
#python $HOME/MedPose/maskrcnn_benchmark/data/datasets/evaluation/poseval/py/evaluate.py --groundTruth=$HOME/MedPose/datasets/posetrack/annotations/val/ --predictions=$SCRATCH/medpose-model-data/models_kprcnn_baseline_res101_w_posetrack/inference/posetrack_val/ --evalPoseEstimation
