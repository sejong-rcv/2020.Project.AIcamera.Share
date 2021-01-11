import numpy as np
import mmcv
import matplotlib.pyplot as plt
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torchvision
from os.path import join, exists
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import random
import os
from os.path import join, exists
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import mmseg

from mmseg.core import mean_iou
import argparse
classes = ('unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump')
    
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
       [0, 11, 123], [118, 20, 12], [122, 81, 25], [255, 88, 255], [211, 22, 53]]
def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')
    parser.add_argument('--data_dir',
                        default='./ir_seg_dataset',
                        help='path to the dataset folder.'
                        )
    parser.add_argument('--model',
                        default='ocr',
                        help='Segmant architecture ocr or psp'
                        )
    parser.add_argument('--input',
                        default='thr',
                        help='input domain thr or rgb'
                        )
    parser.add_argument('--save_path',
                        default='trained',
                        help='save path name'
                        )
    args = parser.parse_args()
    return args
def pprint_result(all_acc, acc, iou, classes):
    summary_str = ''
    summary_str += 'per class results:\n'

    line_format = '{:<15} {:>10} {:>10}\n'
    summary_str += line_format.format('Class', 'IoU', 'Acc')
    class_names = classes
    for i in range(len(class_names)):
        iou_str = '{:.2f}'.format(iou[i] * 100)
        acc_str = '{:.2f}'.format(acc[i] * 100)
        summary_str += line_format.format(class_names[i], iou_str, acc_str)
    summary_str += 'Summary:\n'
    line_format = '{:<15} {:>10} {:>10} {:>10}\n'
    summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

    iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
    acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
    all_acc_str = '{:.2f}'.format(all_acc * 100)
    summary_str += line_format.format('global', iou_str, acc_str,
                                      all_acc_str)
    print(summary_str)
    eval_results = {}
    eval_results['mIoU'] = np.nanmean(iou)
    eval_results['mAcc'] = np.nanmean(acc)
    eval_results['aAcc'] = all_acc

    return eval_results


@DATASETS.register_module()
class MFNet_dataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
if __name__ == '__main__':
    args = return_arguments()
    
    testtype=args.model # 'psp' or 'ocr'

    if testtype=='psp':
        config_file = './configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py'
        checkpoint_file = 'checkpoints/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
    elif testtype=='ocr':
        config_file = './ocrnet_hr48_512x1024_160k_cityscapes.py'
        checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
    """
    PARAMETER SETTING
    """
    trainwith=args.input # thr(thermal) | rgb(rgb) 
    your_base_path=args.data_dir
    
    image_dir='images'
    label_dir='labels'
    rgb_images='split_rgb'
    thr_images='split_thr'
    data_root = your_base_path
    ann_dir = label_dir

    if trainwith=='thr':
        img_dir = thr_images
    elif trainwith=='rgb':
        img_dir = rgb_images


    if not exists(join(your_base_path, 'filesplit')):
        os.makedirs(join(your_base_path, 'filesplit'))
    f=open(join(your_base_path, 'train.txt'), 'r')
    train_images=f.readlines()
    f.close()
    train_images=[i.split('_')[0].split('\n')[0] for i in train_images]
    train_images=list(set(train_images))
    with open(join(your_base_path, 'filesplit', 'train.txt'), 'w') as f:
        for img in train_images:
            f.write(img+'\n')

    f=open(join(your_base_path, 'test.txt'), 'r')
    test_images=f.readlines()
    f.close()
    test_images=[i.split('_')[0].split('\n')[0] for i in test_images]
    test_images=list(set(test_images))
    print('train', len(train_images), '\n', 'test', len(test_images))
    with open(join(your_base_path, 'filesplit', 'val.txt'), 'w') as f:
        for img in test_images:
            f.write(img+'\n')




    from os.path import join

    cfg = Config.fromfile(config_file)
    # Since we use ony one GPU, BN is used instead of SyncBN
    if testtype=='ocr':
        cfg.norm_cfg = dict(type='BN', requires_grad=True)
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head[0].norm_cfg = cfg.norm_cfg
        cfg.model.decode_head[1].norm_cfg = cfg.norm_cfg
        cfg.model.decode_head[0].num_classes = 9
        cfg.model.decode_head[1].num_classes = 9
    elif testtype=='psp':
        cfg.norm_cfg = dict(type='BN', requires_grad=True)
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
        cfg.model.decode_head.num_classes = 9
        cfg.model.auxiliary_head.num_classes = 9

    # Modify dataset type and path
    cfg.dataset_type = 'MFNet_dataset'
    cfg.data_root = your_base_path

    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu=4

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (400, 600)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(480, 640),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]


    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.split = 'filesplit/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'filesplit/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'filesplit/val.txt'

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = checkpoint_file
    # Set up working dir to save files and logs.
    cfg.work_dir = join('./work_dirs/', testtype+'_'+trainwith)

    cfg.total_iters = 10000
    cfg.log_config.interval = 200
    cfg.evaluation.interval = 1000
    cfg.checkpoint_config.interval = 1000

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)


    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmseg.apis import train_segmentor

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    # Add an attribute for visuanlization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                    meta=dict())


    # 정성적 결과 저장을 원한다면 saveImage=True
    saveImage=False # True | False

    label_base=join(your_base_path, label_dir)
    rimage_base=join(your_base_path, rgb_images)
    timage_base=join(your_base_path, thr_images)

    save_path=args.save_path
    r_save_path=save_path+'_with_rgb'
    t_save_path=save_path+'_with_thr'
    if saveImage:
        if not exists(r_save_path):
            os.mkdir(r_save_path)
        if not exists(t_save_path):
            os.mkdir(t_save_path)

    path_name=[]
    gt_seg_maps=[]
    gt_seg_maps_D=[]
    gt_seg_maps_N=[]
    val_results=[]
    val_results_D=[]
    val_results_N=[]
    t_val_results=[]
    t_val_results_D=[]
    t_val_results_N=[]
    for image in tqdm(test_images):
        path_name.append(image)
        image=image+'.png'
        label_pth=join(label_base, image)
        r_image_pth=join(rimage_base, image)
        t_image_pth=join(timage_base, image)
        labelimage=np.asarray(Image.open(label_pth))
        imageimage=np.asarray(Image.open(r_image_pth))
        thermimage=np.asarray(Image.open(t_image_pth))
        result = inference_segmentor(model, imageimage)[0]
        t_result = inference_segmentor(model, thermimage)[0]
        gt_seg_maps.append(labelimage)
        val_results.append(result)
        t_val_results.append(t_result)
        if image[-5]=='N':
            gt_seg_maps_N.append(labelimage)
            val_results_N.append(result)
            t_val_results_N.append(t_result)
        else:
            gt_seg_maps_D.append(labelimage)
            val_results_D.append(result)
            t_val_results_D.append(t_result)




    NUM=len(classes)
    print('### total RGB')
    all_acc, acc, iou = mean_iou(val_results, gt_seg_maps, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))
    print('### Night RGB')
    all_acc, acc, iou = mean_iou(val_results_N, gt_seg_maps_N, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))
    print('### Day RGB')
    all_acc, acc, iou = mean_iou(val_results_D, gt_seg_maps_D, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))

    print('### total Thermal')
    all_acc, acc, iou = mean_iou(t_val_results, gt_seg_maps, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))
    print('### Night Thermal')
    all_acc, acc, iou = mean_iou(t_val_results_N, gt_seg_maps_N, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))
    print('### Day Thermal')
    all_acc, acc, iou = mean_iou(t_val_results_D, gt_seg_maps_D, NUM, ignore_index=255)
    print(pprint_result(all_acc, acc, iou, classes))