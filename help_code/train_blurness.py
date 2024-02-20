import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim

from config_blurness import *
# from backbones import mobilenet_small as model
from backbones import mobilenet_mid_b as model
from utils import init_log
from dataloader.data_loader import blurnessLoader

import numpy as np
import time
from tqdm import tqdm
import argparse
import os
from apex import amp

# export var KMP_DUPLICATE_LIB_OK=TRUE    # 管用

loss_function = nn.MSELoss()
def lr_cos(n):
    lr = 0.5 * (1 + np.cos((n - START_EPOCH) / (EPOCHS - START_EPOCH) * np.pi)) * BASE_LR
    if lr < 1e-6:
        lr = 1e-6
    return lr

def train(data_loader, net, epoch, optimizer, lr):
    net.train()
    train_total_loss = 0.0
    total = 0

    since = time.time()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    tq = tqdm(data_loader, ncols=160, ascii=True)
    for i, data in enumerate(tq):
        img, labels = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer.zero_grad()
        
        outputs = net(img)
        cls_loss = loss_function(outputs, labels.float())
        
        cls_loss.backward()
        # with amp.scale_loss(cls_loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_total_loss += cls_loss.item() * batch_size
        total += batch_size
        tq.set_description(
        (
            f"Train epoch_id:{epoch};"
            f"data_id:{i};"
            f"lr:{lr:.6f};"
            f"loss:{cls_loss:.5f};"
            )
        )

    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = 'Train {} total_loss: {:.4f} lr: {:.6f} time: {:.0f}m {:.0f}s'\
        .format(epoch, train_total_loss, lr, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

def test(data_loader, net, epoch):
    start = time.time()
    net.eval()
    net.cuda()

    test_loss = 0.0 # cost function error
    cnt = 0

    tq = tqdm(data_loader, ncols=160, ascii=True)
    for i, data in enumerate(tq):
        img, labels = data[0].cuda(), data[1].cuda()
        outputs = net(img)

        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        cnt += 1
        tq.set_description(
            (
            f"Test epoch_id:{epoch};"
            f"data_id:{i};"
            f"mse:{loss.item():.5f};"
            )
        )

    finish = time.time()
    _print('Test set: Epoch: {}, Average mse loss: {:.4f}, Time:{:.2f}s'.format(epoch, test_loss/cnt, finish - start))

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
    trainset = blurnessLoader(root=Train_DATA_ROOT, dirpath=Train_DATA_DIR, filepath=Train_DATA_FILE, size=[SIZEH, SIZEW])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS, drop_last=True)

    # val
    valset = blurnessLoader(root=TEST_DATA_ROOT, dirpath=TEST_DATA_DIR, filepath=TEST_DATA_FILE, size=[SIZEH, SIZEW], train=False)
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
