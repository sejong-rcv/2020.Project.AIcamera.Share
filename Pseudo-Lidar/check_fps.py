import argparse
import time
import torch
import numpy as np
import os
import torch.optim as optim

# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader
import os
from PIL import Image
# plot params
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms as transforms
import cv2
from mpl_toolkits.mplot3d import Axes3D

model = get_model("resnet18_md_v2", input_channels=3, pretrained=False)
model=model.to("cuda")

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("models/resnet18_v2_sl1_thermal.pth"))

max_depth = 50;
min_depth = 1;
min_disp  = 1;
def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def disp2depth( disp, max_disp ):
    disp[ disp < min_disp ] = min_disp;
    depth = (3233.93339530 * 0.245) / disp;
    depth[depth < min_depth] = min_depth;
    depth[depth > max_depth] = max_depth;
    return depth
totensor = transforms.ToTensor()
model.eval()
import cv2
import timeit 
with torch.no_grad():
    RGB_image = np.array(Image.open("images/LEFT_000004533.jpg"))
    left_image = Image.open("images/THER_000004533.jpg").convert("RGB")
    left_tensor=totensor(left_image)
    left_tensor = torch.stack((left_tensor, torch.flip(left_tensor, [2])))
    print(left_tensor.shape)
    while 1:
        start_t=timeit.default_timer()
        disps=model(left_tensor.cuda())
        terminated_t=timeit.default_timer()
        FPS=1./(terminated_t-start_t)
        print("FPS : ",FPS)
    