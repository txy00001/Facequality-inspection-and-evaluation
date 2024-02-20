import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim

from config_attribute import *
from backbones import mobilenet_mid as model
from utils import init_log
from dataloader.data_loader import attributeLoader

import numpy as np
import time
from tqdm import tqdm
import argparse
import os
from apex import amp

# export var KMP_DUPLICATE_LIB_OK=TRUE    # 管用

loss_function = nn.CrossEntropyLoss()
def lr_cos(n):
    lr = 0.5 * (1 + np.cos((n - START_EPOCH) / (EPOCHS - START_EPOCH) * np.pi)) * BASE_LR
    if lr < 1e-6:
        lr = 1e-6
    return lr

def train(data_loader, net, epoch, optimizer, lr):
    net.train()
    train_total_loss = 0.0
    total = 0
    loss = 0
    train_total_eye_loss = 0
    train_total_mouth_loss = 0

    since = time.time()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    tq = tqdm(data_loader, ncols=160, ascii=True)
    for i, data in enumerate(tq):

        img, labels_eye, labels_mouth = data[0].cuda(), data[1].cuda(), data[2].cuda()
        batch_size = img.size(0)
        optimizer.zero_grad()
        
        outputs = net(img)
        cls_eye_loss = loss_function(outputs[0], labels_eye)
        cls_mouth_loss = loss_function(outputs[1], labels_mouth)
        
        loss = cls_eye_loss + cls_mouth_loss
        loss.backward()
        # with amp.scale_loss(cls_loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_total_loss += loss.item() * batch_size
        train_total_eye_loss += cls_eye_loss.item() * batch_size
        train_total_mouth_loss += cls_mouth_loss.item() * batch_size
        total += batch_size
        tq.set_description(
        (
            f"Train epoch_id:{epoch};"
            f"data_id:{i};"
            f"lr:{lr:.6f};"
            f"loss:{loss:.5f};"
            f"eye_loss:{cls_eye_loss:.5f};"
            f"mouth_loss:{cls_mouth_loss:.5f};"
            )
        )

    train_total_loss = train_total_loss / total
    train_total_eye_loss = train_total_eye_loss/total
    train_total_mouth_loss = train_total_mouth_loss/total
    time_elapsed = time.time() - since
    loss_msg = 'Train {} total_loss: {:.4f} eye_loss: {:.4f} mouse_loss: {:.4f} lr: {:.6f} time: {:.0f}m {:.0f}s'\
        .format(epoch, train_total_loss, train_total_eye_loss, train_total_mouth_loss, lr, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

@torch.no_grad()
def test(data_loader, net, epoch):
    start = time.time()
    net.eval()
    net.cuda()

    eye_loss = 0.0 # cost function error
    mouth_loss = 0.0 # cost function error
    correct_eye = 0.0
    correct_mouth = 0.0
    cnt = 0

    tq = tqdm(data_loader, ncols=160, ascii=True)
    for i, data in enumerate(tq):
        img, labels_eye, labels_mouth = data[0].cuda(), data[1].cuda(), data[2].cuda()
        outputs = net(img)

        cls_eye_loss = loss_function(outputs[0], labels_eye)
        cls_mouth_loss = loss_function(outputs[1], labels_mouth)
        
        loss = cls_eye_loss + cls_mouth_loss

        eye_loss += cls_eye_loss.item()
        mouth_loss += cls_mouth_loss.item()
        _, preds_eye = outputs[0].max(1)
        _, preds_mouth = outputs[1].max(1)

        correct_eye += (preds_eye.eq(labels_eye)*1.0).sum()
        correct_mouth += (preds_mouth.eq(labels_mouth)*1.0).sum()
        cnt += img.shape[0]
        tq.set_description(
            (
            f"Test epoch_id:{epoch};"
            f"data_id:{i};"
            f"loss:{loss.item():.5f};"
            f"acc_eye:{(preds_eye.eq(labels_eye)*1.0).mean():.5f};"
            f"acc_mouth:{(preds_mouth.eq(labels_mouth)*1.0).mean():.5f};"
            )
        )

    finish = time.time()
    _print('Test set: Epoch: {}, eye_loss: {:.4f}, mouth_loss: {:.4f}, acc_eye: {:.4f}, acc_mouth: {:.4f}, \
        Time:{:.2f}s'.format(epoch, eye_loss/cnt, mouth_loss / cnt, correct_eye/cnt, correct_mouth/cnt, finish - start))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='test')
    args = parser.parse_args()

    # gpu init
    gpu_list = ''
    multi_gpus = False
    if isinstance(GPU, int):
        gpu_list = str(GPU)
    else:
        multi_gpus = True
        for i, gpu_id in enumerate(GPU):
            gpu_list += str(gpu_id)
            if i != len(GPU) - 1:
                gpu_list += ','
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # other init
    start_epoch = START_EPOCH
    save_dir = os.path.join(SAVE_DIR, MODEL_PRE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging = init_log(save_dir)
    _print = logging.info
    _print("="*50)

    # define trainloader and testloader
    trainset = attributeLoader(root=Train_DATA_ROOT, dirpath=Train_DATA_DIR, filepath=Train_DATA_FILE, size=[SIZEH, SIZEW])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS, drop_last=True)

    # val
    valset = attributeLoader(root=TEST_DATA_ROOT, dirpath=TEST_DATA_DIR, filepath=TEST_DATA_FILE, size=[SIZEH, SIZEW], train=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=WORKERS, drop_last=False)

    _print("Train datast: {}".format(Train_DATA_DIR))
    _print("Train dataset nums: {}".format(len(trainset)))
    _print("Val datast: {}".format(TEST_DATA_DIR))
    _print("Val dataset nums: {}".format(len(valset)))
    _print("batch size: {}".format(BATCH_SIZE))

    # define model
    net = model.MobileFaceNet()

    if RESUME:
        ckpt = torch.load(RESUME)
        c = ckpt['net_state_dict']
        model_dict = net.state_dict()
        state_dict = {k:v for k,v in c.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        del model_dict['linear7.conv.weight']
        net.load_state_dict(model_dict, strict=False)
        # start_epoch = ckpt['epoch'] + 1
        # BASE_LR = ckpt['lr']
        _print("load ckpt: {}".format(RESUME))

    if args.test:
        test(valloader, net, epoch=99999)
        exit()

    # define optimizers
    base_params = net.parameters()

    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-4},
    ], lr=BASE_LR, momentum=0.9, nesterov=True)

    # optimizer_ft = optim.Adam([
    #     {'params': base_params, 'weight_decay': 4e-5},
    #     {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
    #     {'params': ArcMargin.weight, 'weight_decay': 4e-4}
    # ], lr=BASE_LR)

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)

    net = net.cuda()
    # net, optimizer_ft = amp.initialize(net, optimizer_ft, opt_level="O0")

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, EPOCHS+1):
        lr = lr_cos(epoch)
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, EPOCHS))
        net.train()
        train(trainloader, net, epoch, optimizer=optimizer_ft, lr=lr)
        
        # save model
        if epoch % SAVE_FREQ == 0:
            msg = 'Saving checkpoint: {}'.format(epoch)
            _print(msg)
            if multi_gpus:
                net_state_dict = net.module.state_dict()
            else:
                net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
        # test model
        if epoch % TEST_FREQ == 0:
            test(valloader, net, epoch)

    print('finishing training')
