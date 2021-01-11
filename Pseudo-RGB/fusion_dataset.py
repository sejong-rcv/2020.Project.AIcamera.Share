from os import listdir
from os.path import isfile, join
from random import sample

import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from image_util import *


class Fusion_Testing_Dataset(Data.Dataset):
    def __init__(self, opt, box_num=8):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.test_img_dir)
        self.IMAGE_DIR = opt.test_img_dir

        self.IMAGE_ID_LIST = [f for f in listdir(self.IMAGE_DIR) if isfile(join(self.IMAGE_DIR, f))]

        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.final_size = opt.fineSize
        self.box_num = box_num

    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.IMAGE_ID_LIST[index].split('.')[0] + '.npz')
        output_image_path = join(self.IMAGE_DIR, self.IMAGE_ID_LIST[index])
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path, self.box_num)
        
        img_list = []
#         pil_img = read_to_pil(output_image_path)

        _, pil_img = gen_gray_color_pil(output_image_path)
        img_list.append(self.transforms(pil_img))
        
        cropped_img_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
        for i in index_list:
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size))
            box_info_2x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 2))
            box_info_4x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 4))
            box_info_8x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, self.final_size // 8))
            cropped_img = self.transforms(pil_img.crop((startx, starty, endx, endy)))
            cropped_img_list.append(cropped_img)
        output = {}
        output['full_img'] = torch.stack(img_list)
        output['file_id'] = self.IMAGE_ID_LIST[index].split('.')[0]
        if len(pred_bbox) > 0:
            output['cropped_img'] = torch.stack(cropped_img_list)
            output['box_info'] = torch.from_numpy(box_info).type(torch.long)
            output['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
            output['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
            output['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
            output['empty_box'] = False
        else:
            output['empty_box'] = True
        return output

    def __len__(self):
        return len(self.IMAGE_ID_LIST)


class Training_Full_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff
    '''
    def __init__(self, opt):

        self.is_ther = opt.input_ther

        if self.is_ther:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.IMAGE_DIR_THER = opt.train_thermal_img_dir

            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.THER_ID_LIST = [f for f in listdir(self.IMAGE_DIR_THER) if isfile(join(self.IMAGE_DIR_THER, f))]
            self.RGB_ID_LIST.sort()
            self.THER_ID_LIST.sort()
        else:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.RGB_ID_LIST.sort()

        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                            transforms.ToTensor()])
        
    def __getitem__(self, index):

        if self.is_ther:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            THER_image_path = join(self.IMAGE_DIR_THER, self.THER_ID_LIST[index])
            gt_img, input_img = gen_ther_color_pil(RGB_image_path, THER_image_path)# output : rgb_img, ther_img
            
        else:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            gt_img, input_img = gen_gray_color_pil(RGB_image_path)# output : rgb_img, gray_img
    
        output = {}
        output['gt_img'] = self.transforms(gt_img)
        output['input_img'] = self.transforms(input_img)
            
        return output

    def __len__(self):

        return len(self.RGB_ID_LIST)


class Training_Instance_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using inference_bbox.py

    It would be better if you can filter out the images which don't have any box.
    '''
    def __init__(self, opt):
        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.train_color_img_dir)

        self.is_ther = opt.input_ther

        if self.is_ther:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.IMAGE_DIR_THER = opt.train_thermal_img_dir

            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.THER_ID_LIST = [f for f in listdir(self.IMAGE_DIR_THER) if isfile(join(self.IMAGE_DIR_THER, f))]

            self.RGB_ID_LIST.sort()
            self.THER_ID_LIST.sort()
        else:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.RGB_ID_LIST.sort()
        self.transforms = transforms.Compose([
            transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        
        pred_info_path = join(self.PRED_BBOX_DIR, self.RGB_ID_LIST[index].split('.')[0] + '.npz')

        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path)

        if self.is_ther:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            THER_image_path = join(self.IMAGE_DIR_THER, self.THER_ID_LIST[index])
            gt_img, input_img = gen_ther_color_pil(RGB_image_path, THER_image_path)# output : rgb_img, ther_img

        else:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            gt_img, input_img = gen_gray_color_pil(RGB_image_path)# output : rgb_img, gray_img

        index_list = range(len(pred_bbox))
        index_list = sample(index_list, 1)
        startx, starty, endx, endy = pred_bbox[index_list[0]]

        output = {}
        output['gt_img'] = self.transforms(gt_img)
        output['input_img'] = self.transforms(input_img)

        return output

    def __len__(self):
        return len(self.RGB_ID_LIST)


class Training_Fusion_Dataset(Data.Dataset):
    '''
    Training on COCOStuff dataset. [train2017.zip]
    
    Download the training set from https://github.com/nightrome/cocostuff

    Make sure you've predicted all the images' bounding boxes using inference_bbox.py

    It would be better if you can filter out the images which don't have any box.
    '''
    def __init__(self, opt, box_num=8):

        self.PRED_BBOX_DIR = '{0}_bbox'.format(opt.train_color_img_dir)

        self.is_ther = opt.input_ther

        if self.is_ther:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.IMAGE_DIR_THER = opt.train_thermal_img_dir

            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.THER_ID_LIST = [f for f in listdir(self.IMAGE_DIR_THER) if isfile(join(self.IMAGE_DIR_THER, f))]
            self.RGB_ID_LIST.sort()
            self.THER_ID_LIST.sort()

        else:
            self.IMAGE_DIR_RGB = opt.train_color_img_dir
            self.RGB_ID_LIST = [f for f in listdir(self.IMAGE_DIR_RGB) if isfile(join(self.IMAGE_DIR_RGB, f))]
            self.RGB_ID_LIST.sort()
            
        self.transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2),
                                              transforms.ToTensor()])
        self.final_size = opt.fineSize
        self.box_num = box_num

    def __getitem__(self, index):
        pred_info_path = join(self.PRED_BBOX_DIR, self.RGB_ID_LIST[index].split('.')[0] + '.npz')
        pred_bbox = gen_maskrcnn_bbox_fromPred(pred_info_path, self.box_num)

        full_gt_list = []
        full_input_list = []

        if self.is_ther:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            THER_image_path = join(self.IMAGE_DIR_THER, self.THER_ID_LIST[index])
            gt_img, input_img = gen_ther_color_pil(RGB_image_path, THER_image_path)# output : rgb_img, ther_img
            
        else:
            RGB_image_path = join(self.IMAGE_DIR_RGB, self.RGB_ID_LIST[index])
            gt_img, input_img = gen_gray_color_pil(RGB_image_path)# output : rgb_img, gray_img

        full_gt_list.append(self.transforms(gt_img))
        full_input_list.append(self.transforms(input_img))
        
        cropped_gt_list = []
        cropped_input_list = []
        index_list = range(len(pred_bbox))
        box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
        for i in range(len(index_list)):
            startx, starty, endx, endy = pred_bbox[i]
            box_info[i] = np.array(get_box_info(pred_bbox[i], gt_img.size, self.final_size))
            box_info_2x[i] = np.array(get_box_info(pred_bbox[i], gt_img.size, self.final_size // 2))
            box_info_4x[i] = np.array(get_box_info(pred_bbox[i], gt_img.size, self.final_size // 4))
            box_info_8x[i] = np.array(get_box_info(pred_bbox[i], gt_img.size, self.final_size // 8))
            cropped_gt_list.append(self.transforms(gt_img.crop((startx, starty, endx, endy))))
            cropped_input_list.append(self.transforms(input_img.crop((startx, starty, endx, endy))))

        output = {}
        output['cropped_gt'] = torch.stack(cropped_gt_list)
        output['cropped_input'] = torch.stack(cropped_input_list)

        output['full_gt'] = torch.stack(full_gt_list)
        output['full_input'] = torch.stack(full_input_list)

        output['box_info'] = torch.from_numpy(box_info).type(torch.long)
        output['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
        output['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
        output['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
        output['file_id'] = self.RGB_ID_LIST[index]
        
        return output

    def __len__(self):
        return len(self.RGB_ID_LIST)