#!/bin/sh


export CUDA_VISIBLE_DEVICES=0,1,2,3


export PYTHONPATH=./
python train.py  --config=config/nyu_pspnet18.yaml