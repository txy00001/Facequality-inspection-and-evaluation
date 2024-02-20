import os
import sys
import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
# import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import transforms, datasets
import torchvision

from torch.utils.data import DataLoader, Dataset
import torch
import glob
import numpy as np
from sklearn import preprocessing
import cv2

"""def txy_rgb_transform(rgb_img):
    out_image = rgb_img

    #高斯滤波
    gblur_kernel=(5,5)
    out_image= cv2.GaussianBlur(out_image,gblur_kernel,0)
    #亮度拉升
    scale = np.random.uniform(0.2,1)
    out_image = out_image*scale
    out_image = out_image.astype(np.uint8)

    return out_image """


class ClassifyDataset(data_utils.Dataset):
    def __init__(self, root_path, data_file, img_size=128):
        self.data_files = np.loadtxt(data_file, dtype=np.str)
        # print(self.data_files)
        self.root_path = root_path
        self.transforms = torchvision.transforms.Compose([
            transforms.RandomApply([transforms.RandomResizedCrop(128, scale=(0.95, 1), ratio=(1, 1))]),
            torchvision.transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, item):
        data_file = self.data_files[item]
        data_file = data_file.split(sep='|', maxsplit=-1)
        # image_file_path=data_file[0].split(sep='/', maxsplit=-1)[4]
        # image_file=os.path.join(self.root_path,image_file_path)

        image_file_path = data_file[0]
        image_file = os.path.join(self.root_path, image_file_path)
        img = Image.open(image_file)
        img = self.transforms(img)
        # print(img.shape)
        """ img=cv2.imread(image_file)
        img=boyan_rgb_transform(img)
        img=cv2.resize(img,(256,256)) 
        img=img.reshape(3,256,256)"""

        blurness = torch.tensor([float(data_file[1])]).type(torch.FloatTensor)###模糊度
        # if float(data_file[4])==3:
        #     mouth=1
        # else:
        #     mouth=0
        # leftEye=int(data_file[2])
        # rightEye=int(data_file[3])
        #########嘴巴#############
        # if float(data_file[4]) == 0:###带着口罩
        #     mouth = 1
        # else:
        #     mouth = 0
        # #  ##############  左眼##############
        # leftEye = int(data_file[2])
        # if leftEye == 0 or leftEye == 2:  # 睁眼
        #     leftEye = 0
        # elif leftEye == 1 or leftEye == 3:  # 闭眼
        #     leftEye = 1
        # elif leftEye == 4:  ####带着墨镜
        #     leftEye = 2
        # elif leftEye == 5:  #####其他
        #     leftEye = 3
        #
        # ########右眼############
        # rightEye = int(data_file[3])
        # if  rightEye == 0 or rightEye == 2:  # 睁眼
        #     rightEye = 0
        # elif  rightEye == 1 or  rightEye == 3:  # 闭眼
        #     rightEye = 1
        # elif rightEye == 4:  ####带着墨镜
        #     rightEye = 2
        # elif leftEye == 5:  #####其他
        #     rightEye = 3
        #


        dict_data = {
            'img': img,
            'labels': {
                'blurness_labels': blurness,
                # 'left_eye_labels': leftEye,
                # 'right_eye_labels': rightEye,
                # 'mouth_labels': mouth
            }
        }

        return dict_data

    def __len__(self):
        return len(self.data_files)

# def __len__(self):
#     return len(self.data_files)


if __name__ == '__main__':
    root_path = '/opt/data/share/zhenghanfei/dataset/IQIYI/iQIYI-VID-FACE/'
    data_file = './data/train_new.txt'
    train_dataset = ClassifyDataset(root_path, data_file)
    print(train_dataset.__getitem__(0))
    # train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
