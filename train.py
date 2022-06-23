import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os
from datetime import datetime
import torch.nn as nn
from model.R_models import R_RES
from data import get_loader
from utils import clip_gradient, adjust_lr
from save_log import *
from loss.edge_SSIM_loss import SSIM
from loss.DICE import DICE
from loss.WBCE import WBCE
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keys = 'R_Model'
best_acc = 1
time_ = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))


class Parser:
    def __init__(self):
        self.epoch = 65
        self.lr = 1e-4
        self.momen = 0.9
        self.decay = 5e-4
        self.batchsize = 9
        self.trainsize = 352
        self.clip = 0.5
        self.is_gtNet = False
        self.decay_rate = 0.1
        self.decay_epoch = 50
        self.using_SGD = False
        self.weight_decay = 1e-5
        self.model = {'R_Model': R_RES}
        print('epoch:', self.epoch, 'learning rate:', self.lr, 'decay_epoch:', self.decay_epoch, 'decay_rate:',
              self.decay_rate, 'using SGD:', self.using_SGD, 'train size:', self.trainsize, 'batch size',
              self.batchsize, 'weight_decay', self.weight_decay)


class LOSS():

    def __init__(self):
        super(LOSS, self).__init__()
        self.loss = [WBCE(), SSIM(), DICE()]

    def __call__(self, x, y):
        loss = 0
        for i in range(len(self.loss)):
            loss += self.loss[i](x, y)
        return loss


def validate(data_loader, model, optimizer, best_acc, epoch):
    '''model.eval()
    acces1 = []
    acces2 = []
    steps = len(data_loader)
    with torch.no_grad():
        for i, pack in enumerate(data_loader):
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()
            # optimizer.zero_grad()
            atts, dets = model(images)
            acc1 = 0
            acc2 = 0
            gt = gts.data.cpu().numpy().squeeze()

            at = atts.sigmoid().data.cpu().numpy().squeeze()
            at = (at - at.min()) / (at.max() - at.min() + 1e-20)

            dt = dets.sigmoid().data.cpu().numpy().squeeze()
            dt = (dt - dt.min()) / (dt.max() - dt.min() + 1e-20)
            acc1 += 1 - np.mean(np.abs(at - gt))
            acc2 += 1 - np.mean(np.abs(dt - gt))
            acces1.append(acc1)
            acces2.append(acc2)
            if (i + 1) % 100 == 0 or (i + 1) == len(data_loader):
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], acc1: {:.4f}, acc2: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i + 1, steps, np.mean(acces1), np.mean(acces2)))'''

    acc = 1  # np.mean(acces2)
    if acc > best_acc:
        best_acc = acc
        state = {
            'model': model.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch,
            'opt': optimizer.state_dict()
        }
        torch.save(state, os.path.join('./models', keys, time_, 'model_best_acc.pth'))
    if epoch >= 60:
        state = {
            'model': model.state_dict(),
            'best_acc': best_acc,
            'epoch': epoch,
            'opt': optimizer.state_dict()
        }
        torch.save(state, os.path.join('./models', keys, time_, 'model_%d.pth' % epoch))
    return best_acc, acc


def train(train_loader, model, optimizer, epoch, loss):
    model.train()
    loss1s = []
    loss2s = []
    for i, pack in enumerate(train_loader):
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)

        images = images.cuda()
        gts = gts.cuda()

        optimizer.zero_grad()
        atts, dets = model(images)

        loss1 = loss(atts.sigmoid(), gts)
        loss2 = loss(dets.sigmoid(), gts)
        losses = loss1 + loss2
        losses.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss1s.append(loss1.item())
        loss2s.append(loss2.item())
        if (i + 1) % 200 == 0 or (i + 1) == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i + 1, total_step, np.mean(loss1s), np.mean(loss2s)))

    return sum(loss1s) / len(loss1s), sum(loss2s) / len(loss2s)


iter = 0
if __name__ == '__main__':
    model_name = '2021_12_01_19_02_34'
    if iter > 0:
        time_ = model_name
    # order = int(sys.argv[1])
    if not os.path.exists(os.path.join('./models', keys, time_)):
        os.makedirs(os.path.join('./models', keys, time_))
    best_acc = 0
    best_epoch = 0
    CE = torch.nn.BCELoss()  # BCEWithLogitsLoss=sigmoid+BCELoss
    loss = LOSS()

    sys.stdout = Logger('./log/train_' + keys + time_ + '.txt')
    if iter > 0:
        time_ = model_name

    opt = Parser()
    best_loss = 1e26
    # build models

    model = opt.model[keys]()
    model.cuda()
    if torch.cuda.device_count() >= 2:
        print("num GPUs: ", torch.cuda.device_count())
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = torch.nn.DataParallel(model).to(device)
    if iter > 0:
        dic = torch.load(os.path.join('./models', keys, model_name, 'model_best_loss.pth'))
        model.load_state_dict(dic['model'])
        best_loss = dic["best_loss"]
        best_acc = dic["best_acc"]
        best_epoch = dic['epoch']
        base_h, base_p, head1_p, head2_p = [], [], [], []
        for name, param in model.named_parameters():
            if 'resnet' in name:
                if 'layer' in name:
                    base_p.append(param)
                else:
                    base_h.append(param)
            elif 'head1' in name:
                head1_p.append(param)
            else:
                head2_p.append(param)
        optimizer = torch.optim.Adam([{'params': base_h}, {'params': base_p}, {'params': head1_p}, {'params': head2_p}],
                                     opt.lr,weight_decay=opt.weight_decay)
        optimizer.load_state_dict(dic['opt'])
        iter = best_epoch + 1
        print(
            'best epoch:%d, best loss:%.4f,best acc:%.4f' % (best_epoch,best_loss,best_acc))
        del dic
    else:
        base_h, base_p, head1_p, head2_p = [], [], [], []
        for name, param in model.named_parameters():
            if 'resnet' in name:
                if 'layer' in name:
                    base_p.append(param)
                else:
                    base_h.append(param)
            elif 'head1' in name:
                head1_p.append(param)
            else:
                head2_p.append(param)
        optimizer = torch.optim.Adam([{'params': base_h}, {'params': base_p}, {'params': head1_p}, {'params': head2_p}],
                                     opt.lr,weight_decay=opt.weight_decay)
    train_loader = get_loader('./data/DUTS-TR/DUTS-TR-Image/', './data/DUTS-TR/DUTS-TR-Mask/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize, mode='train', shuffle=True)
    test_loader = get_loader('./data/DUTS-TE/DUTS-TE-Image/', './data/DUTS-TE/DUTS-TE-Mask/',
                             batchsize=opt.batchsize,
                             trainsize=opt.trainsize, mode='val', shuffle=False)
    total_step = len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True,threshold=0)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.epoch,last_epoch=iter-1)
    print("Let's go!")
    adjust_lr(optimizer, opt.lr, 0, opt.decay_rate, opt.decay_epoch)
    '''if iter == 0:
        adjust_lr(optimizer, opt.lr, 0, opt.decay_rate, opt.decay_epoch)
    else:
        lr_scheduler.step(best_loss)'''
    for epoch in range(iter, opt.epoch):
        #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        loss1, loss2 = train(train_loader, model, optimizer, epoch, loss)
        best_acc, acc = validate(test_loader, model, optimizer, best_acc, epoch)
        if loss2 < best_loss:
            best_loss = loss2
            state = {
                'model': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
                'epoch': epoch,
                'opt': optimizer.state_dict()
            }
            torch.save(state, os.path.join('./models', keys, time_, 'model_best_loss.pth'))
        if best_acc == acc:
            best_epoch = epoch
        print('{} Epoch [{:03d}/{:03d}], Loss1: {:.4f} Loss2: {:0.4f} acc: {:0.4f} best_acc: {:0.4f} best_epoch: {:}'.
              format(datetime.now(), epoch, opt.epoch, loss1, loss2, acc, best_acc, best_epoch))
        lr_scheduler.step(loss2)
