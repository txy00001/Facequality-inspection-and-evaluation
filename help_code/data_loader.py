import numpy as np
import scipy.misc
import os
import torch
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F
import imgaug.augmenters as iaa
import dlib
from imutils import face_utils, translate, rotate, resize
from PIL import Image

class blurnessLoader(object):
    def __init__(self, root, dirpath='', filepath='', size=[128, 128], train=True):
        self.root = root
        self.dirpath = dirpath
        self.filepath = filepath
        if isinstance(dirpath,list):
            assert len(dirpath) == len(filepath)
        for ii in range(len(dirpath)):
            fp = self.filepath[ii]
            dp = self.dirpath[ii]
            img_txt_file = os.path.join(root, fp)
            self.image_list = []
            self.label_list = []
            self.sizeW = size[1]
            self.sizeH = size[0]
            self.train = train
            with open(img_txt_file) as f:
                img_label_list = f.read().splitlines()
            for info in img_label_list:
                image_name, label_name = info.split('|')[0:2]
                self.image_list.append(os.path.join(root, dp, image_name))
                self.label_list.append(float(label_name))
        self.image_iaa = iaa.Sequential([
             # geometric
            iaa.Fliplr(0.5), # Horizontal Flip
            iaa.Flipud(0.05),
            iaa.Sometimes(0.5, iaa.Rot90(1, keep_size=False)),
            # iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.Affine(
                # scale={"x": (0.9, 1.01), "y": (0.9, 1.01)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-5, 5),
                shear=(-6, 6),
                order=[0, 1],
                # cval=(0, 255),
                mode=['constant']
            ),
            # iaa.CropToAspectRatio(aspect_ratio, position='uniform'),
            iaa.CropAndPad(percent=(-0.1, 0.01), pad_mode=['constant']),
            # iaa.ElasticTransformation(alpha=10, sigma=10), # 局部位移像素
            iaa.Resize({"height": self.sizeH, "width": self.sizeW}),
        ], random_order=True)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = cv2.imread(img_path)
        
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
        return img, label

    def __len__(self):
        return len(self.image_list)


class attributeLoader(object):
    def __init__(self, root, dirpath='CASIA-WebFace-112X96', filepath='CASIA-WebFace-112X96.txt', size=[128, 128], train=True):
        self.root = root
        self.dirpath = dirpath
        self.filepath = filepath
        if isinstance(dirpath,list):
            assert len(dirpath) == len(filepath)
        for ii in range(len(dirpath)):
            fp = self.filepath[ii]
            dp = self.dirpath[ii]
            img_txt_file = os.path.join(root, fp)
            self.image_list = []
            self.label_eye_list = []
            self.label_mouth_list = []
            self.sizeW = size[1]
            self.sizeH = size[0]
            self.train = train
            with open(img_txt_file) as f:
                img_label_list = f.read().splitlines()
            for info in img_label_list:
                image_info = info.split('|')
                # imageName|blurness|leftEyeType|rightEyeType|mouthType|pitch|yaw|roll
                # 0-不戴眼镜睁眼 1-不戴眼镜闭眼 2-戴眼镜睁眼 3-戴眼镜闭眼 4-戴墨镜 5-其他遮挡
                image_name = image_info[0]
                image_info[2:5] = [int(image_info[i]) for i in range(2, 5)]
                if (image_info[2] == 0 or image_info[2] == 2) and (image_info[3] == 0 or image_info[3] == 2):   # 睁眼
                    eye_label = 0
                elif image_info[2] == 1 or image_info[2] == 3 or image_info[3] == 1 or image_info[3] == 3: # 闭眼
                    eye_label = 1
                elif (image_info[2] == 4 or image_info[3] == 4):
                    eye_label = 2
                else:
                    eye_label = 3
                # 0-口罩  1-其他遮挡  2-闭嘴  3-张嘴,目前iqiyi中其他遮挡大多数是无遮挡不戴口罩，分类精度很差现在先归为无遮挡
                if image_info[4] == 0:
                    mouth_label = 1
                # elif image_info[4] == 1:
                #     mouth_label = 2
                else:
                    mouth_label = 0
                self.image_list.append(os.path.join(root, dp, image_name))
                self.label_eye_list.append(int(eye_label))
                self.label_mouth_list.append(int(mouth_label))

        self.dark_glass_list = []
        self.xr_glass_list = []
        self.eye_glass_list = []
        self.getGlassData()

        self.mask_list = []
        self.getMaskData()
        
        self.image_iaa = iaa.Sequential([
            # geometric
            iaa.Fliplr(0.5), # Horizontal Flip
            # iaa.Flipud(0.05),
            # iaa.Sometimes(0.5, iaa.Rot90(1, keep_size=False)),
            # iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
            iaa.Affine(
                # scale={"x": (0.9, 1.01), "y": (0.9, 1.01)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
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
            # iaa.AddToHueAndSaturation((-40, 40), per_channel=True),
            # iaa.Sometimes(0.2, iaa.ChangeColorTemperature((1100, 10000))),

            iaa.Sometimes(0.2, iaa.GammaContrast((0.5, 2.0), per_channel=True)),
            # iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
            iaa.Sometimes(0.5, iaa.color.Grayscale(alpha=[0.0, 0.8])),
            # imgcorruptlike
            # iaa.Sometimes(0.01, iaa.imgcorruptlike.Fog(severity=[1, 2])),
            # iaa.Sometimes(0.05, iaa.imgcorruptlike.JpegCompression(severity=[1, 2])),
            iaa.Sometimes(0.15, iaa.imgcorruptlike.GaussianNoise(severity=[1])),
            # iaa.Sometimes(0.15, iaa.imgcorruptlike.ShotNoise(severity=[1])),
            # iaa.Sometimes(0.15, iaa.imgcorruptlike.SpeckleNoise(severity=[1])),
            # pillike
            # iaa.Sometimes(0.15, iaa.pillike.EnhanceSharpness(factor=[0.5, 1.5])),
            # iaa.Sometimes(0.05, iaa.pillike.EnhanceColor(factor=[0.5, 1.5])),
            # iaa.CoarseDropout(0.04, size_percent=0.03, per_channel=1),
            iaa.Resize({"height": self.sizeH, "width": self.sizeW}),
        ], random_order=True)

        self.det_face = dlib.get_frontal_face_detector()
        self.det_landmarks = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # 68点

    def getData(self):
        cnt = 0
        image_list = []
        label_list = []
        for ii in range(len(self.dirpath)):
            img_txt_file = os.path.join(self.root, self.filepath[ii])
            classID = 0
            with open(img_txt_file) as f:
                img_label_list = f.read().splitlines()
            for info in img_label_list:
                image_dir, label_name = info.split(' ')
                if 'vgg' in self.dirpath[ii]:
                    image_list.append(os.path.join(self.root, self.dirpath[ii], image_dir+'.jpg'))
                else:
                    image_list.append(os.path.join(self.root, self.dirpath[ii], image_dir))
                label_list.append(int(label_name)+cnt)
                classID = max(int(label_name), classID)
            cnt += classID + 1
        return image_list, label_list

    def getGlassData(self):
        glass_root = '/opt/data/private/dataset/glass_sample/'
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
        img=resize(image, width=500)
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
        shades_width =  shape[16, 0] - shape[0, 0]
        current_deal = glass.resize((shades_width, int(shades_width * glass.size[1] / glass.size[0])),
                            resample=Image.LANCZOS)
        current_deal = current_deal.rotate(angle, expand=True)
        current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 以两眼间中心为中点
        left_x = shape[27, 0] - shades_width//2
        left_y = shape[19, 1] # 右眉毛最上边

        img.paste(current_deal, (left_x, left_y), current_deal)   #调节眼镜位置
        img.resize((width, height))
        return np.array(img)[..., ::-1], 1

    def getMaskData(self):
        glass_root = '/opt/data/private/dataset/mask/mask_sample/mask_rgba'
        mask_dir = ''
        mask_files = os.listdir(os.path.join(glass_root, mask_dir))
        self.mask_list = [os.path.join(glass_root, mask_dir, dark_file) for dark_file in mask_files]

    def genMask(self, image, mask):
        height, width, c = image.shape
        img=resize(image, width=500)
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
        shades_width =  shape[16, 0] - shape[0, 0]
        shades_height = shape[8, 1] - shape[27, 1]
        # current_deal = mask.resize((shades_width, int(shades_width * mask.size[1] / mask.size[0])),
        #                     resample=Image.LANCZOS)
        current_deal = mask.resize((shades_width, shades_height), resample=Image.LANCZOS)
        current_deal = current_deal.rotate(angle, expand=True)
        current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 以鼻梁上的28点为中心点
        left_x = shape[29, 0] - shades_width//2
        left_y = shape[28, 1]

        img.paste(current_deal, (left_x, left_y), current_deal)
        img.resize((width, height))
        return np.array(img)[..., ::-1], 1

    def __getitem__(self, index):
        img_path = self.image_list[index]
        eye_label = self.label_eye_list[index]
        mouth_label = self.label_mouth_list[index]
        img = cv2.imread(img_path)

        if eye_label < 2:
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
                # elif glass_rand < 0.2:
                #     rand_idx = np.random.randint(0, len(self.eye_glass_list))
                #     glass_path = self.eye_glass_list[rand_idx]
                #     # glass_img = Image.open(glass_path) #RGBA   --->log打印有问题
                #     glass = cv2.imread(glass_path, -1)
                #     glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
                #     glass_img = Image.fromarray(glass)
                #     img = self.gen_glass(img, glass_img)
                else:
                    rand_idx = np.random.randint(0, len(self.xr_glass_list))
                    glass_path = self.xr_glass_list[rand_idx]
                    # glass_img = Image.open(glass_path) #RGBA   --->log打印有问题
                    glass = cv2.imread(glass_path, -1)
                    glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
                    glass_img = Image.fromarray(glass)
                    img, flag = self.genGlass(img, glass_img)
                if flag:
                    eye_label = 2

        if mouth_label == 0:
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
                    mouth_label = 1

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
        return img, eye_label, mouth_label

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import time
    # ============test CASIA_Face==========
    # data_dir = '/home/xjsd/zhf/dataset/FaceRecognition'
    # dataset = CASIA_Face(root=data_dir)
    # trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    # print(len(dataset))
    # for data in trainloader:
    #     print(data[0].shape)

    # ===========test FaceLoader===========
    Train_DATA_ROOT = '/opt/data/private/dataset'

    IQIYI_DATA_DIR = 'IQIYI/iQIYI-VID-FACE/'
    IQIYI_DATA_FILE = 'IQIYI/face_attributes_btrain.txt'

    Train_DATA_DIR = [IQIYI_DATA_DIR]
    Train_DATA_FILE = [IQIYI_DATA_FILE]

    # Test
    TEST_DATA_ROOT = '/opt/data/private/dataset'

    TEST_IQIYI_DATA_DIR = 'IQIYI/iQIYI-VID-FACE/'
    TEST_IQIYI_DATA_FILE = 'IQIYI/face_attributes_bval.txt'

    TEST_DATA_DIR = [TEST_IQIYI_DATA_DIR]
    TEST_DATA_FILE = [TEST_IQIYI_DATA_FILE]

    t1 = time.time()
    # dataset = blurnessLoader(TEST_DATA_ROOT, TEST_DATA_DIR, TEST_DATA_FILE, size=[128, 128])
    dataset = attributeLoader(Train_DATA_ROOT, Train_DATA_DIR, Train_DATA_FILE, size=[128, 128], train=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    print(len(dataset))

    for idx, data in enumerate(trainloader):
        print(idx)
        print(data[1], "  ", data[2])
        if idx > 100:
            exit()
    #     a = 0
    #     d = data[0].cuda()
    #     print(idx)
    #     if idx > 10000:
    #         break

    # t2 = time.time()
    # print(f"耗时: {(t2-t1)/60} min")
