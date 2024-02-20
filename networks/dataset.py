import os
import sys
import glob
import imgaug.augmenters as iaa
from cv2 import resize
from imutils import face_utils
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
#import pandas as pd

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
import dlib
from imutils import face_utils, translate, rotate, resize


class ClassifyDataset(data_utils.Dataset):
    def __init__(self,root_path,data_file,img_size=128,train=True):
        # self.data_files=np.loadtxt(data_file,dtype=np.str)
        self.data_files = data_file
        self.root_path = root_path

        self.transforms = torchvision.transforms.Compose([
            transforms.RandomApply([transforms.RandomResizedCrop(128, scale=(0.95, 1), ratio=(1, 1))]),
            torchvision.transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        for ii in range(len(data_file)):
            df = self.data_files[ii]
            # fp = self.filepath[ii]
            # dp = self.dirpath[ii]
            img_txt_file = os.path.join(root_path, df)
            self.image_list = []
            self.label_eye_list = []
            self.label_mouth_list = []
            self.label_blurness_list=[]
            self.sizeW = 128
            self.sizeH = 128
            self.train = train
            with open(img_txt_file) as f:  #####
                img_label_list = f.read().splitlines()
            for info in img_label_list:
                image_info = info.split('|')
                # imageName|blurness|leftEyeType|rightEyeType|mouthType|pitch|yaw|roll
                # 0-不戴眼镜睁眼 1-不戴眼镜闭眼 2-戴眼镜睁眼 3-戴眼镜闭眼 4-戴墨镜 5-其他遮挡
                image_name = image_info[0]
                blurness =image_info[1]

                image_info[2:5] = [int(image_info[i]) for i in range(2, 5)]
                if (image_info[2] == 0 or image_info[2] == 2) and (image_info[3] == 0 or image_info[3] == 2):  # 睁眼
                    eye = 0
                elif (image_info[2] == 1 or image_info[2] == 3) and (image_info[3] == 1 or image_info[3] == 3):  # 闭眼
                    eye = 1
                elif (image_info[2] == 4 or image_info[3] == 4):
                    eye = 2
                else:
                    eye = 3
                # 0-口罩  1-其他遮挡  2-闭嘴  3-张嘴,目前iqiyi中其他遮挡大多数是无遮挡不戴口罩，分类精度很差现在先归为无遮挡
                if image_info[4] == 0:
                    mouth = 1
                # elif image_info[4] == 1:
                #     mouth_label = 2
                else:
                    mouth = 0
                self.image_list.append(os.path.join(root_path, df, image_name))
                self.label_eye_list.append(int(eye))
                self.label_mouth_list.append(int(mouth))
                self.label_blurness_list.append(blurness)



        self.dark_glass_list = []
        self.xr_glass_list = []
        self.eye_glass_list = []
        self.getGlassData()

        self.mask_list = []
        self.getMaskData()

        self.image_iaa = iaa.Sequential([
            # geometric
            iaa.Fliplr(0.5),  # Horizontal Flip

            iaa.Affine(
                rotate=(-5, 5),
                shear=(-6, 6),
                order=[0, 1],
                # cval=(0, 255),
                mode=['constant']
            ),
            # iaa.CropToAspectRatio(aspect_ratio, position='uniform'),
            iaa.CropAndPad(percent=(-0.1, 0.01), pad_mode=['constant']),

            # blur
            iaa.Sometimes(0.05, iaa.MotionBlur(k=[5, 10])),
            # color
            iaa.Sometimes(0.5, iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),

            iaa.Sometimes(0.2, iaa.GammaContrast((0.5, 2.0), per_channel=True)),
            iaa.Sometimes(0.5, iaa.color.Grayscale(alpha=[0.0, 0.8])),
            iaa.Sometimes(0.15, iaa.imgcorruptlike.GaussianNoise(severity=[1])),
            iaa.Resize({"height": self.sizeH, "width": self.sizeW}),
        ], random_order=True)

        self.det_face = dlib.get_frontal_face_detector()
        self.det_landmarks = dlib.shape_predictor("/new_face_class/shape_predictor_68_face_landmarks.dat")  # 68点

    def getData(self):
        cnt = 0
        data_files = []
        label_list = []
        for ii in range(len(self.data_files)):
            img_txt_file = os.path.join(self.root_path, self.data_files[ii])
            classID = 0
            with open(img_txt_file) as f:
                img_label_list = f.read().splitlines()
            for info in img_label_list:
                image_dir, label_name = info.split(' ')
                if 'vgg' in self.data_files[ii]:
                    data_files.append(os.path.join(self.root_path, self.data_files[ii], image_dir + '.jpg'))
                else:
                    data_files.append(os.path.join(self.root_path, self.data_files[ii], image_dir))
                label_list.append(int(label_name) + cnt)
                classID = max(int(label_name), classID)
            cnt += classID + 1
        return data_files, label_list

    def getGlassData(self):
        glass_root = '/opt/data/share/zhenghanfei/dataset/glass_sample'
        dark_glass_dir = 'dark_glass/'
        xr_glass_dir = 'xr_glass/'
        eye_glass_dir = 'eye_glass/'
        dark_files = os.listdir(os.path.join(glass_root, dark_glass_dir))
        xr_files = os.listdir(os.path.join(glass_root, xr_glass_dir))
        eye_files = os.listdir(os.path.join(glass_root, eye_glass_dir))

        self.dark_glass_list = [glass_root + dark_glass_dir + dark_file for dark_file in dark_files]
        self.xr_glass_list = [glass_root + xr_glass_dir + xr_file for xr_file in xr_files]
        self.eye_glass_list = [glass_root + eye_glass_dir + eye_file for eye_file in eye_files]

    def genGlass(self, image, glass):
        height, width, c = image.shape
        img = resize(image, width=500)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 用于检测当前人脸所在位置方向的预测器
        rects = self.det_face(img_gray, 0)
        if len(rects) == 0:
            return image, 0
        rect = rects[0]

        # shades_width = rect.right() - rect.left()
        shape = self.det_landmarks(img_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 从输入图像中抓取每只眼睛的轮廓
        leftEye = shape[36:42]
        rightEye = shape[42:48]

        # 计算每只眼睛的中心
        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        # 计算眼心之间的夹角
        dY = leftEyeCenter[1] - rightEyeCenter[1]
        dX = leftEyeCenter[0] - rightEyeCenter[0]
        angle = np.rad2deg(np.arctan2(dY, dX))

        # 图片重写
        shades_width = shape[16, 0] - shape[0, 0]
        current_deal = glass.resize((shades_width, int(shades_width * glass.size[1] / glass.size[0])),
                                    resample=Image.LANCZOS)
        current_deal = current_deal.rotate(angle, expand=True)
        current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

        # 以两眼间中心为中点
        left_x = shape[27, 0] - shades_width // 2
        left_y = shape[19, 1]  # 右眉毛最上边

        img.paste(current_deal, (left_x, left_y), current_deal)  # 调节眼镜位置
        img.resize((width, height))
        return np.array(img)[..., ::-1], 1

    def getMaskData(self):
        mask_root = '/opt/data/share/zhenghanfei/dataset/mask/mask_sample'
        mask_dir = 'mask_rgba/'
        mask_files = os.listdir(os.path.join(mask_root, mask_dir))
        self.mask_list = [os.path.join(mask_root, mask_dir, dark_file) for dark_file in mask_files]

    def genMask(self, image, mask):
        height, width, c = image.shape
        img = resize(image, width=500)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 用于检测当前人脸所在位置方向的预测器
        rects = self.det_face(img_gray, 0)
        if len(rects) == 0:
            return image, 0
        rect = rects[0]

        # shades_width = rect.right() - rect.left()
        shape = self.det_landmarks(img_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 计算脸部最大轮廓之间的夹角
        dY = shape[0, 1] - shape[16, 1]
        dX = shape[0, 0] - shape[16, 0]
        angle = np.rad2deg(np.arctan2(dY, dX))

        # 图片重写
        shades_width = shape[16, 0] - shape[0, 0]
        shades_height = shape[8, 1] - shape[27, 1]
        # current_deal = mask.resize((shades_width, int(shades_width * mask.size[1] / mask.size[0])),
        #                     resample=Image.LANCZOS)
        current_deal = mask.resize((shades_width, shades_height), resample=Image.LANCZOS)
        current_deal = current_deal.rotate(angle, expand=True)
        current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

        # 以鼻梁上的28点为中心点
        left_x = shape[29, 0] - shades_width // 2
        left_y = shape[28, 1]

        img.paste(current_deal, (left_x, left_y), current_deal)
        img.resize((width, height))
        return np.array(img)[..., ::-1], 1

    def __getitem__(self, index):
        # data_file = self.data_files[item]
        # data_file = data_file.split(sep='|', maxsplit=-1)
        # image_file_path = data_file[0]
        # image_file = os.path.join(self.root_path, image_file_path)
        # img = Image.open(image_file)
        # img = self.transforms(img)

        img_path = self.image_list[index]
        eye = self.label_eye_list[index]
        mouth = self.label_mouth_list[index]
        blurness = self.label_blurness_list[index]
        img = cv2.imread(img_path)

        if eye < 2:
            # 一定概率贴眼镜：
            glass_rand = np.random.uniform()
            if glass_rand < 0.4:
                if glass_rand < 0.2:
                    rand_idx = np.random.randint(0, len(self.dark_glass_list))
                    glass_path = self.dark_glass_list[rand_idx]
                    # glass_img = Image.open(glass_path) #RGBA   --->log打印有问题
                    glass = cv2.imread(glass_path, -1)
                    glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
                    glass_img = Image.fromarray(glass)
                    img, flag = self.genGlass(img, glass_img)

                else:
                    rand_idx = np.random.randint(0, len(self.xr_glass_list))
                    glass_path = self.xr_glass_list[rand_idx]
                    # glass_img = Image.open(glass_path) #RGBA   --->log打印有问题
                    glass = cv2.imread(glass_path, -1)
                    glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
                    glass_img = Image.fromarray(glass)
                    img, flag = self.genGlass(img, glass_img)
                if flag:
                    eye = 2

        if mouth == 0:
            mask_rand = np.random.uniform()
            if mask_rand < 0.4:
                rand_idx = np.random.randint(0, len(self.mask_list))
                glass_path = self.mask_list[rand_idx]
                # glass_img = Image.open(glass_path) #RGBA   --->log打印有问题
                glass = cv2.imread(glass_path, -1)
                glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
                glass_img = Image.fromarray(glass)
                img, flag = self.genMask(img, glass_img)
                if flag:
                    mouth = 1

        img = cv2.resize(img, [self.sizeW, self.sizeH])
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = img[:, :, ::-1]

        if self.train:
            img = self.image_iaa(images=[img])[0]
        else:
            img = cv2.resize(img, [self.sizeW, self.sizeH])
        img = img / 127.5 - 1.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()


        dict_data = {
            'img': img,
            'labels': {
                'blurness_labels': blurness,
                'eye_labels': eye,
                'mouth_labels': mouth,
            }
        }

        return dict_data


    def __len__(self):
        return len(self.data_files)



if __name__ == '__main__':
    root_path='/opt/data/share/zhenghanfei/dataset/IQIYI/iQIYI-VID-FACE/'
    data_file='/new_face_class/data/train_new.txt'


    train_dataset = ClassifyDataset(root_path,data_file)
    print(train_dataset.__getitem__(0))
    #train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
