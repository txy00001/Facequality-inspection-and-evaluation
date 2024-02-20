import os
import sys
import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from networks.dataset import ClassifyDataset
from torch.utils.data import DataLoader, Dataset

from networks.new_net import MobileFaceNet


from train import calculate_metrics,checkpoint_load,get_loss


def validate(model, dataloader,device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        step=0

        for batch in dataloader:
            step+=1
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))
            print('案例:',step,'预测值：',output['blurness'].detach().cpu().numpy(),output['eye'].cpu().max(1)[1].detach().cpu().numpy(),output['mouth'].cpu().max(1)[1].detach().cpu().numpy())
            print('案例:',step,'真实值：',target_labels['blurness_labels'].detach().cpu().numpy(),target_labels['eye_labels'].detach().cpu().numpy(),target_labels['mouth_labels'].detach().cpu().numpy())
            if step==50:
                break

           


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model= DAN()
    model= MobileFaceNet()
    root_path='/opt/data/share/zhenghanfei/dataset/IQIYI/iQIYI-VID-FACE/'
    train_data_file='./data/train_new.txt'
    val_data_file='./data/val_new.txt'
    train_dataset = ClassifyDataset(root_path,train_data_file)
    train_loader = DataLoader(train_dataset, batch_size=512,shuffle=True)
    val_dataset = ClassifyDataset(root_path,val_data_file)
    val_dataloader = DataLoader(val_dataset, batch_size=512,shuffle=True)
    print(len(train_loader),len(val_dataloader))
    validate(model, val_dataloader, device,checkpoint='/new_face_class/checkpoints/2022-12-16_22-57/checkpoint-000112.pth')


        