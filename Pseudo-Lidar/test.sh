#!/usr/bin/env bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5 python main_monodepth_pytorch.py --mode test \
        --model resnet50_md --model_path models/Ori_last.pth --output_directory ./Ori_v2
