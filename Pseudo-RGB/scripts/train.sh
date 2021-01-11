COLOR_DATASET_DIR=./datas/MTN/traindata/RGB
THER_DATASET_DIR=./datas/MTN/traindata/THER
# Stage 1: Training Full Image Colorization
mkdir ./checkpoints/full
cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/full/

python train.py --stage full --name full --input_ther --sample_p 1.0 --niter 100 \
                --niter_decay 50 --load_model --lr 0.0005 --model train --fineSize 256 \
                --batch_size 16 --display_ncols 3 --display_freq 1600 --print_freq 1600 \
                --train_color_img_dir $COLOR_DATASET_DIR --train_thermal_img_dir $THER_DATASET_DIR --gpu_ids 1,2,3

# # # Stage 2: Training Instance Image Colorization
mkdir ./checkpoints/instance
cp ./checkpoints/full/latest_net_G.pth ./checkpoints/instance/
python train.py --stage instance --name instance --input_ther --sample_p 1.0 --niter 100 \
                --niter_decay 50 --load_model --lr 0.0005 --model train --fineSize 256\
                --batch_size 16 --display_ncols 3 --display_freq 1600 --print_freq 1600\
                --train_color_img_dir $COLOR_DATASET_DIR --train_thermal_img_dir $THER_DATASET_DIR --gpu_ids 1,2,3

# # Stage 3: Training Fusion Module
mkdir ./checkpoints/mask
cp ./checkpoints/full/latest_net_G.pth ./checkpoints/mask/latest_net_GF.pth
cp ./checkpoints/instance/latest_net_G.pth ./checkpoints/mask/latest_net_G.pth
cp ./checkpoints/full/latest_net_G.pth ./checkpoints/mask/latest_net_GComp.pth
python train.py --stage fusion --name mask --sample_p 1.0 --input_ther --niter 10 --niter_decay 20 \
                --lr 0.00005 --model train --load_model --display_ncols 4 --fineSize 256 \
                --batch_size 1 --display_freq 500 --print_freq 500 --train_color_img_dir $COLOR_DATASET_DIR\
                --train_thermal_img_dir $THER_DATASET_DIR --gpu_ids 2,3