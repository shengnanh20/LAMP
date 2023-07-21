#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -c 10
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python3 tools/train.py --cfg experiments/crowdpose.yaml --gpus 0,1,2,3