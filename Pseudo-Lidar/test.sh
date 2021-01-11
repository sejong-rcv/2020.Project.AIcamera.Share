#!/usr/bin/env bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 python main_monodepth_pytorch.py --model resnet18_md  \
       --model_path model_output/res18_RGB/resnet18-5c106cde_cpt.pth --mode test --RGB True \
       --output_directory v1_RGB_l1
       

