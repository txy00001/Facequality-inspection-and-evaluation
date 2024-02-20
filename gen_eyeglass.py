import cv2
import dlib
import os
from PIL import Image
from imutils import face_utils, translate, rotate, resize
import numpy as np
import random

# 创建人脸检测器
det_face = dlib.get_frontal_face_detector()
# 加载标志点检测器
det_landmarks = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")  # 68点

root = r'D:\xjsd\project\insight\images'
save_root = r'D:\xjsd\project\insight\images'
if not os.path.exists(save_root):
    os.mkdir(save_root)

def gen_glass(image, glass):
    height, width, c = image.shape
    img=resize(image, width=500)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 用于检测当前人脸所在位置方向的预测器
    rects = det_face(img_gray, 0)
    if len(rects) == 0:
        return image
    rect = rects[0]
    x = rect.left()
    y = rect.top() #could be face.bottom() - not sure
    w = rect.right() - rect.left()
    h = rect.bottom() - rect.top()

    # shades_width = rect.right() - rect.left()
    shape = det_landmarks(img_gray, rect)
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
                           resample=Image.Resampling.LANCZOS)
    current_deal = current_deal.rotate(angle, expand=True)
    current_deal = current_deal.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # 左上点的位置调节
    # left_x = int(shape[0, 0] - shades_width * 0.15)   # 脸轮廓
    # left_y = shape[19, 1] # 眉毛的最上边
    
    # 以两眼间中心为中点
    left_x = shape[27, 0] - shades_width//2
    left_y = shape[19, 1]

    img.paste(current_deal, (left_x, left_y), current_deal)   #调节眼镜位置
    # cv2.imshow('image', np.array(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
    img.resize((width, height))
    return np.array(img)[..., ::-1]


def single_image():
    # glass = Image.open(r'D:\Dataset\glass_sample\xr_glass\08.png')
    glass = cv2.imread(r'D:\Dataset\glass_sample\xr_glass\08.png', -1)
    glass = cv2.cvtColor(glass, cv2.COLOR_BGRA2RGBA)
    glass = Image.fromarray(glass)
    image = cv2.imread(r'D:\xjsd\project\insight\images\zhf02.jpg')
    glass_image = gen_glass(image, glass)
    # cv2.imwrite(r'D:\xjsd\project\insight\images\zhf02-glass.jpg', glass_image)
    cv2.imshow('image', glass_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

def files_image():
    glass_root = r'D:\Dataset\test_FAR\FF\search_false\CALFW'
    save_root = r'D:\Dataset\test_FAR\FF\xr_glass\search_false\CALFW_5100'
    glass_list = os.listdir(glass_root)
    glass_list = [os.path.join(glass_root, glass_path) for glass_path in glass_list]
    # glass = Image.open(r'D:\Dataset\glass_sample\xr_glass\08.png')
    glass = cv2.imread(r'D:\Dataset\glass_sample\xr_glass\08.png', -1)
    glass = Image.fromarray(glass)
    for file in glass_list:
        print(file)
        image = cv2.imread(file)
        glass_image = gen_glass(image, glass)
        # cv2.imwrite(os.path.join(save_root, os.path.basename(file)), glass_image)
        cv2.imshow('image', glass_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

if __name__ == '__main__':

    single_image()
    files_image()
    # glass = Image.open(r'D:\Dataset\glass_sample\eye_glass\07.png')#.convert("RGBA") 
    glass_root = r'D:\Dataset\glass_sample\eye_glass'
    glass_list = os.listdir(glass_root)
    glass_list = [os.path.join(glass_root, glass_path) for glass_path in glass_list]

    for rt, dirs, files in os.walk(root):
        for dir in dirs:
            print(dir)
            if dir == 'CALFW_noregister' or dir == 'tmp':
                continue
            if not os.path.exists(os.path.join(save_root, dir)):
                os.mkdir(os.path.join(save_root, dir))
            for r, ds, fs in os.walk(os.path.join(root, dir)):
                for f in fs:
                    print(f)
                    glass_index = random.randint(0, len(glass_list)-1)
                    glass = Image.open(glass_list[glass_index])

                    image = cv2.imread(os.path.join(r, f))
                    glass_image = gen_glass(image, glass)
                    # cv2.imwrite(os.path.join(save_root, dir, f), glass_image)
                    # cv2.imshow('image', glass_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit()

                

