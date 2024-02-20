import os
import sys
import glob
from tqdm import tqdm


from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import Parameter
from torchvision import transforms, datasets
from networks.dataset import ClassifyDataset
from torch.utils.data import DataLoader, Dataset

from networks.new_net import MobileFaceNet

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
import warnings
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm
from networks.loss import MultiClassFocalLossWithAlpha,GHM_Loss,WBCEWithLogitLoss,DSCLoss,MultiFocalLoss,DiceLoss

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def get_loss(net_output, ground_truth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focalloss_eye=MultiClassFocalLossWithAlpha(alpha=[0.15, 0.15 ,0.3]).to(device)

    focalloss_mouth=nn.CrossEntropyLoss()  
    SmoothL1Loss_blus=nn.SmoothL1Loss()
    eye_loss = focalloss_eye(net_output['eye'], ground_truth['eye_labels'])

    mouth_loss = focalloss_mouth(net_output['mouth'], ground_truth['mouth_labels'])
    blurness_loss = SmoothL1Loss_blus(net_output['blurness'], ground_truth['blurness_labels'])
    loss = eye_loss + mouth_loss + blurness_loss
    return loss, {'eye': eye_loss, 'mouth': mouth_loss,'blurness': blurness_loss}

def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)

def calculate_metrics(output, target):
    #print(output['eye'].size(),target['eye_labels'].size(),output['mouth'].size(),target['mouth_labels'].size(),output['blurness'].size(),target['blurness_labels'].size())
    _, predicted_eye = output['eye'].cpu().max(1)
    gt_eye = target['eye_labels'].cpu()

    _, predicted_mouth = output['mouth'].cpu().max(1)
    gt_mouth = target['mouth_labels'].cpu()

    predicted_blurness = output['blurness'].cpu()
    gt_blurness = target['blurness_labels'].cpu()
    cmu_MSE=torch.nn.MSELoss()
    mse_blurness=cmu_MSE(predicted_blurness,gt_blurness)
    with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
        warnings.simplefilter("ignore")
        accuracy_eye = balanced_accuracy_score(y_true=gt_eye.numpy(), y_pred=predicted_eye.numpy())
        accuracy_mouth = balanced_accuracy_score(y_true=gt_mouth.numpy(), y_pred=predicted_mouth.numpy())
        #accuracy_blurness = balanced_accuracy_score(y_true=gt_blurness.numpy(), y_pred=predicted_blurness.numpy())

    return accuracy_eye, accuracy_mouth,mse_blurness


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='gpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

@torch.no_grad()
def validate(model, dataloader,  iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_Eye = 0

        accuracy_mouth = 0
        mse_blurness=0
        for batch in tqdm(val_dataloader,total=len(val_dataloader)):
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            img=img.to(device)
            img=img.type(torch.cuda.FloatTensor)
            output = model(img)

            val_train, val_train_losses = get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_Eye, batch_accuracy_mouth, batch_mse_blurness = \
                calculate_metrics(output, target_labels)

            accuracy_Eye += batch_accuracy_Eye
            accuracy_mouth += batch_accuracy_mouth
            mse_blurness += batch_mse_blurness

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_Eye /= n_samples

    accuracy_mouth /= n_samples
    mse_blurness /= n_samples
    print("Validation  loss: {:.4f}, blurness: {:.4f}, Eye: {:.4f},  mouth: {:.4f}\n".format(
        avg_loss, mse_blurness ,accuracy_Eye, accuracy_mouth))



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model= DAN()
    model= MobileFaceNet()
    pre_train=True
    if pre_train==True:
        # pretext_model = torch.load('checkpoint-000134.pth')
        pretext_model = torch.load('new_face_class/checkpoints/2022-12-18_01-12/checkpoint-000500.pth')
        model2_dict = model.state_dict()
        state_dict = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)


    # pretrain= torch.load('weights/045_ms1m.ckpt')
    # if pretrain is not None:
    #     ckpt = torch.load(pretrain)
    #     if "state_dict" in ckpt:
    #         model.load_state_dict(ckpt["state_dict"])
    #     else:"""  """
    #         model.model.load_state_dict(ckpt)


    model=model.to(device)
    root_path='/opt/data/share/zhenghanfei/dataset/IQIYI/iQIYI-VID-FACE/'
    train_data_file='./data/train_new.txt'
    val_data_file='./data/val_new.txt'

    train_dataset = ClassifyDataset(root_path,train_data_file)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True, num_workers=16,drop_last=True)
    val_dataset = ClassifyDataset(root_path,val_data_file, train=False)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256,shuffle=True, num_workers=16,drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256,shuffle=False, num_workers=16,drop_last=False)
    print(len(train_dataloader),len(val_dataloader))

    optimizer = torch.optim.Adam(model.parameters())
    logdir = os.path.join('./logs/', get_cur_time())
    # savedir = os.path.join('./checkpoints/', get_cur_time())
    savedir = os.path.join('./weights/checkpoint-000134.pth')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    n_train_samples = len(train_dataloader)

    ## debug
    validate(model, val_dataloader, 0, device)
    exit()

    print("Starting training ...")

    for epoch in range(1, 500 + 1):
        total_loss = 0
        accuracy_Eye = 0

        accuracy_mouth = 0
        mse_blurness=0
        for batch in tqdm(train_dataloader,total=len(train_dataloader)):
            optimizer.zero_grad()
            
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            img=img.to(device)
            img=img.type(torch.cuda.FloatTensor)
            output = model(img)
            
            loss_train, losses_train = get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_Eye,  batch_accuracy_mouth, batch_mse_blurness = \
                calculate_metrics(output, target_labels)

            accuracy_Eye += batch_accuracy_Eye
            accuracy_mouth += batch_accuracy_mouth
            mse_blurness += batch_mse_blurness
            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f},blurness: {:.4f} ,Eye: {:.4f}, mouth: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            mse_blurness / n_train_samples,
            accuracy_Eye / n_train_samples,

            accuracy_mouth / n_train_samples))

        if epoch % 1 == 0:
            validate(model, val_dataloader, epoch, device)

        if epoch % 5 == 0:
            checkpoint_save(model, savedir, epoch)
