###EVALUATION 
import os
import cv2
import numpy as np
from PIL import Image
import PIL
import torch
from IQA_pytorch import SSIM, utils, LPIPSvgg
from tqdm import tqdm
import math


def PSNR(predict, gt):
    predict = np.asarray(predict)
    gt = np.asarray(gt)
    
    mse = np.mean((predict-gt)**2)
    if mse ==0:
        return 100
    PIXEL_MAX = 255.0
    
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

data_path = os.path.join('datas','MTN','testdata','RGB')
predict_path = os.path.join('results_ther2rgb')
if __name__ == '__main__':
    
    gt_list = os.listdir(data_path)
    gt_list.sort()
    predict_list = os.listdir(predict_path)
    predict_list.sort()

    ssim_score = 0
    psnr_score = 0
    lpips_score = 0

    for name in tqdm(predict_list):

        img_name = name.split('.png')[0]
        try:
            index = gt_list.index('LEFT'+img_name.split('THER')[1]+'.jpg')
            # index = gt_list.index(name)
        except:
            continue
        
        gt = Image.open(os.path.join(data_path,gt_list[index]))
        gt = gt.resize((256,256))

        predict = Image.open(os.path.join(predict_path,name))
        psnr_score += PSNR(predict, gt)

        gt = utils.prepare_image(gt).cuda()
        predict = utils.prepare_image(predict).cuda()

        model1 = SSIM(channels=3)
        model2 = LPIPSvgg(channels=3).cuda()

        ssim_score += model1(predict, gt, as_loss=False)
        lpips_score += model2(predict, gt, as_loss=False)

    print('avg_ssim_score: %.4f' % (ssim_score/len(predict_list)).item())
    print('avg_psnr_score: %.4f' %(psnr_score/len(predict_list)))
    print('avg_lpips_score: %.4f'%(lpips_score/len(predict_list)))
    
