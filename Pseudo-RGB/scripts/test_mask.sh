INPUT_DIR=example

# INPUT_DIR=example 
OUTPUT_DIR=results_ther2rgb
mkdir results_ther2rgb
cp ./checkpoints/instance/latest_net_G.pth ./checkpoints/mask/latest_net_G.pth
cp ./checkpoints/full/latest_net_G.pth ./checkpoints/mask/latest_net_GComp.pth

python inference_bbox.py --test_img_dir $INPUT_DIR --dataset MTN
CUDA_VISIBLE_DEVICES=7 python test_fusion.py --name test_fusion --sample_p 1.0 \
                                             --model fusion --fineSize 256 \
                                             --test_img_dir $INPUT_DIR --results_img_dir $OUTPUT_DIR