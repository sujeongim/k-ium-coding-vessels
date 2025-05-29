#!/bin/bash
unset LD_LIBRARY_PATH
# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(dirname $(pwd))":"$(pwd)"

# 학습 실행
python train.py --loss='combined'
