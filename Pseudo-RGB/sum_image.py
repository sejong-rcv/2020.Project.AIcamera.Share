import subprocess
import os
from os import listdir
from os.path import join, isfile, isdir
from subprocess import call
from tqdm import tqdm
import cv2
from PIL import Image
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

DATASET_DIR='/raid/datasets/MTN_data'
# DATASET_DIR = '/raid/datasets/New_Sejong_RCV_dataset/RGBTDv4/'

txt_list = [f for f in listdir(join(DATASET_DIR,'txt')) if isfile(join(join(DATASET_DIR,'txt'), f))]


if args.train:
    save_path = join('datas','train')
    txt_path = [p for p in txt_list if p=='train.txt']
else:
    save_path = join('datas','test')
    txt_path = [p for p in txt_list if p=='test.txt']

if os.path.isdir(join(save_path,'THER')) is False:
    print('Create path: {0}'.format(join(save_path,'THER')))
    os.makedirs(join(save_path,'THER'))

if os.path.isdir(join(save_path,'RGB')) is False:
    print('Create path: {0}'.format(join(save_path,'RGB')))
    os.makedirs(join(save_path,'RGB'))

f = open(join(join(DATASET_DIR,'txt'), txt_path[0]), mode='rt')

for i,line in enumerate(tqdm(f)):
#     #potenit
#     img_list = line.split('/')
#     if img_list[-1][-1] == '\n':
#         ther = join(img_list[0],img_list[1],'Ther',img_list[-1][:-1])
#         rgb = join(img_list[0],img_list[1],'RGB',img_list[-1][:-1])
#     else:
#         ther = join(img_list[0],img_list[1],'Ther',img_list[-1])
#         rgb = join(img_list[0],img_list[1],'RGB',img_list[-1])
    
    # MTN
    img_list = line.split(' ')
    rgb = img_list[0]
    ther = img_list[-1][:-1]

    call(['cp','-p',join(DATASET_DIR,ther),join(save_path,'THER')])
    call(['cp','-p',join(DATASET_DIR,rgb), join(save_path,'RGB')])
