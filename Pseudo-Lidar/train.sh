#!/usr/bin/env bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=6,7 python main_monodepth_pytorch.py --model resnet18_md --l_type l1 --model_output_directory model_output/res18_RGB --RGB True
