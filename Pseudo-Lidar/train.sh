#!/usr/bin/env bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3 python main_monodepth_pytorch.py  \
        --model resnet18_md --model_path models/resnet18_md_RGB 
