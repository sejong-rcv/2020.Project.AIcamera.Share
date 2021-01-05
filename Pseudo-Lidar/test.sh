#!/usr/bin/env bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python main_monodepth_pytorch.py --mode test \
        --model resnet18_md --model_path models/Ori_last.pth --output_directory ./Ori
