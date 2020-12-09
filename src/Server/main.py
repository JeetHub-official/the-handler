
import sys
import json
import numpy as np
from tqdm import tqdm
from numpy import uint8
from flask import abort

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import random
import math
import numbers
import collections
import numpy as np
import pandas as pd
import torch
import cv2
import scipy.ndimage
from PIL import Image, ImageOps
import pandas as pd
try:
    import accimage
except ImportError:
    accimage = None

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import mobilenetv2

from opts import parse_opts

from spatial_transforms import *
from temporal_transforms import *

from torchvision import transforms
from torchvision.transforms import ToPILImage
import time

import zipfile
import os
import shutil

from multiprocessing import set_start_method
from utils import *




def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]
 
def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def intialize(model):
    global opt
    if not opt.no_cuda:
        #print('GPU CUDA')
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        if opt.pretrain_path:
            #print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:

                model.module.classifier = nn.Sequential(
                    nn.Dropout(0.9),
                    nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                )
                
                model.module.classifier = model.module.classifier.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        #print('CPU')
        if opt.pretrain_path:
            #print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path,map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'],strict=False)

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:

                model.classifier = nn.Sequential(
                    nn.Dropout(0.9),
                    nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes)
                )

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, model.parameters()



def val_epoch(epoch, data_loader, model, criterion, opt):
    #print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            if not opt.no_cuda:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())

        if opt.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[0])
        else:
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(targets, 1)[0])

        prec1, prec3 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        top1.update(prec1, inputs.size(0))
        top3.update(prec3, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                'Prec@3 {top3.val:.5f} ({top3.avg:.5f})'.format(
                    epoch,
                    i,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top3=top3))
        

    opt.top1_values_val.append(f'{top1.avg:.5f}')
    opt.top3_values_val.append(f'{top3.avg:.5f}')
    opt.val_loss_values.append(f'{losses.avg:.5f}')
    

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt):
    
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())

        if opt.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[0])
        else:
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(targets, 1)[0])
            
        losses.update(loss.data, inputs.size(0))
        prec1, prec3 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        top1.update(prec1, inputs.size(0))
        top3.update(prec3, inputs.size(0))

        optimizer.zero_grad()
        
        if opt.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'            
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@3 {top3.val:.5f} ({top3.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top3=top3,
                      lr=optimizer.param_groups[0]['lr']))
        
        
    opt.top1_values.append(f'{top1.avg:.5f}')
    opt.top3_values.append(f'{top3.avg:.5f}')
    opt.train_loss_values.append(f'{losses.avg:.5f}')


def train(model,parameters,training_data,validation_data):
    global opt

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)
    """
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=8, #check this
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)
    """
    
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    #print(model)

    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)


    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        print('At epoch:',i)
        if not opt.no_train:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt)
            #val_epoch(i, val_loader, model, criterion, opt)

    return model

def pre_process_train(list_of_frames):
    global opt
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    
    spatial_transform = Compose([
        crop_method,
        ToTensor(opt.norm_value)
    ])
    spatial_transform.randomize_parameters()

    clip = [spatial_transform(Image.fromarray(cv2.cvtColor(img.astype(uint8),cv2.COLOR_BGR2RGB))) for img in list_of_frames]

    im_dim = clip[0].size()[-2:]

    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

    return clip

