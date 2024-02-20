import os

import cv2
import numpy as np
import torch
from PIL.Image import Image
from cv2 import resize
from imutils import face_utils

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
                image_list.append(os.path.join(self.root, self.dirpath[ii], image_dir + '.jpg'))
            else:
                image_list.append(os.path.join(self.root, self.dirpath[ii], image_dir))
            label_list.append(int(label_name) + cnt)
            classID = max(int(label_name), classID)
        cnt += classID + 1
    return image_list, label_list


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
    glass_root = '/opt/data/share/zhenghanfei/dataset/mask/mask_sample/mask_rgba'
    mask_dir = ''
    mask_files = os.listdir(os.path.join(glass_root, mask_dir))
    self.mask_list = [os.path.join(glass_root, mask_dir, dark_file) for dark_file in mask_files]


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
    img_path = self.image_list[index]
    eye = self.label_eye_list[index]
    mouth = self.label_mouth_list[index]
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
    return img, eye, mouth


def __len__(self):
    return len(self.image_list)

def data_screen(file):
    f=np.loadtxt(file,dtype=np.str_)
    #print('原先数据集大小',len(f))
    train=[]
    for j in range(2,4):
        count0=0
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        for i in range(len(f)):
            a=f[i].split(sep='|', maxsplit=-1)#不限拆分次数
            #print(a)
            if float(a[j])==0:
                # count+=1
                # train.append(f[i])
                count0 += 1
                if count0 < 400000:  # 类型选择xxx个
                    train.append(f[i])
            elif float(a[j])==1:
                count1+=1
                if count1 < 400000:
                 train.append(f[i])
            elif float(a[j])==2:
                count2+=1
                if count2 < 400000:
                 train.append(f[i])
            elif float(a[j])==3:
                count3+=1
                if count3 < 400000:
                    train.append(f[i])#此处数量太小全用了

                train.append(f[i])
            elif float(a[j])==4:
                count4+=1
                train.append(f[i])

            elif float(a[j])==5:
                count5 += 1
                if count5 < 400000:
                    train.append(f[i])

    np.savetxt("./data/face_test3.txt", train,fmt='%s',delimiter=' ')


#统计各类别的数量
def cal_class_num(file):
    f = np.loadtxt(file, dtype=np.str_)
    print('数据集大小', len(f))
    count_clear = 0
    count_blur = 0
    for k in range(len(f)):
        a = f[k].split(sep='|', maxsplit=-1)
        if float(a[1]) < 0.7:
            count_clear += 1
        else:
            count_blur += 1
    print('模糊各类别数', count_clear+count_blur)

    for j in range(2,6):
        count0=0
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        for i in range(len(f)):
            a=f[i].split(sep='|', maxsplit=-1)
            #print(a)
            if float(a[j])==0:
                count0+=1
            elif float(a[j])==1:
                count1+=1
            elif float(a[j])==2:
                count2+=1
            elif float(a[j])==3:
                count3+=1
            elif float(a[j])==4:
                count4+=1
            elif float(a[j])==5:
                count5+=1
        if j==2:
            print('左眼各类别数:',count0+count2,count1+count3,count4,count5)
        elif j==3:
            print('右眼各类别数:',count0+count2,count1+count3,count4,count5)
        elif j==4:
            print('嘴巴各类别数:',count0,count1+count2+count3)
            print('*'*70)


#数据集划分
def Data_division(file):
    f=np.loadtxt(file,dtype=np.str_)
    train=[]
    val=[]
    #数据集划分为8：2
    for i in range(len(f)):
        if i%6==0:
            val.append(f[i])
        else:
            train.append(f[i])
    np.savetxt("./data/val_new.txt", val,fmt='%s',delimiter=' ')
    np.savetxt("./data/train_new.txt", train,fmt='%s',delimiter=' ')
    print('训练集大小：',np.array(train).shape,'测试集大小：',np.array(val).shape)



if __name__ == '__main__':
    #统计筛选后的各类别数大小，根据自己数据调整
    file='./data/new_face_data.txt'
    cal_class_num(file)

    #首先筛选数据，大致保证各类别数相差不要太大，需根据自己数据调整
    file='./data/new_face_data.txt'#筛选后保存在face_test1.txt中
    data_screen(file)

    #统计筛选后的各类别数大小，根据自己数据调整
    file='./data/face_test3.txt'
    cal_class_num(file)

    #数据集划分，分为测试集和训练集，保存在对应train.txt和val.txt中
    file='./data/face_test3.txt'
    Data_division(file)
