from sklearn.utils.extmath import density
import torch
import time
import os
from pathlib import Path
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
from torchvision.utils import save_image
from pytorchyolo import detect, models
from pytorchyolo import train
from pytorchyolo.utils.utils import to_cpu, load_classes, rescale_boxes, non_max_suppression, xywh2xyxy, box_iou
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.test import _create_data_loader, _create_validation_data_loader
from pytorchyolo.train import _create_train_data_loader
from module import VAE
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
import cv2
import numpy as np
import glob


# # Get data configuration
# data_config = parse_data_config("data/custom.data")
# op_data_path = data_config['train']

# with open(op_data_path) as f:
#     lines = f.readlines()

# lines = [line[:-1] for line in lines]

lines = sorted(os.listdir('kde_plot'))
lines = ['kde_plot/'+line for line in lines]

img_array = []
for filename in lines:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('kde.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
  