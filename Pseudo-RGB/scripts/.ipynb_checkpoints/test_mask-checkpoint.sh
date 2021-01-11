INPUT_DIR=datas/potenit/testdata/THER

# INPUT_DIR=example 
OUTPUT_DIR=results_thergray2rgb_ft
mkdir results_thergray2rgb_ft
# cp ./checkpoints/coco_instance/latest_net_G.pth ./checkpoints/coco_mask/latest_net_G.pth
# cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GComp.pth

python inference_bbox.py --test_img_dir $INPUT_DIR
CUDA_VISIBLE_DEVICES=7 python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir $INPUT_DIR --results_img_dir $OUTPUT_DIR