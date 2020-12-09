
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


def addGuassNoise(img):
    row,col,ch= img.shape
    mean = 0.2
    var = 0.3
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy.astype(uint8)
  
def transform_data_aug(img, transform):
    
    if transform=='':
        return img
    elif transform=='Blur':
        return cv2.blur(img,(10,10))
    elif transform=='Brightness':
        return (img + (150/255)).astype(uint8)
    elif transform=='Contrast':
        return (img * 0.5).astype(uint8)
    elif transform=='AddNoise':
        return addGuassNoise(img)
    elif transform=='CentreCrop':
        h, w, _ = img.shape
        margin = 5
        x1, y1, x2, y2 = h*margin/100, w*margin/100, h-h*margin/100, w-w*margin/100 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[x1:x2,y1:y2]
    elif transform=='RightUPCrop':
        h, w, _ = img.shape
        margin = 10
        x1, y1, x2, y2 = 0, w*margin/100, h-h*margin/100, w 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[x1:x2,y1:y2]
    elif transform=='RightDOWNCrop':
        h, w, _ = img.shape
        margin = 10
        x1, y1, x2, y2 = h*margin/100, w*margin/100, h, w
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[x1:x2,y1:y2]
    elif transform=='LeftUPCrop':
        h, w, _ = img.shape
        margin = 10
        x1, y1, x2, y2 = 0, 0, h-h*margin/100, w-w*margin/100
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[x1:x2,y1:y2]
    elif transform=='LeftDOWNCrop':
        h, w, _ = img.shape
        margin = 10
        x1, y1, x2, y2 = h*margin/100, 0, h, w-w*margin/100 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return img[x1:x2,y1:y2]
    else:
        return img

def getData(id, directory):
    global opt
    
    print('Extracting data')
    os.mkdir(directory)
    with zipfile.ZipFile(f'data{id}.zip',"r") as zip_ref:
        zip_ref.extractall(directory)
    
    os.remove(f'data{id}.zip')
    df=pd.read_csv(f'{directory}/class_labels.csv', sep=',',header=None)
    labels = df.values
    print('Labels Detected:',labels)

    transforms_for_data_aug = ['','Blur','Brightness','Contrast', 'AddNoise', 'CentreCrop', 'RightUPCrop', 'RightDOWNCrop', 'LeftUPCrop', 'LeftDOWNCrop']
 
    train_dataset = [] 
    val_dataset = []
    files = {}  
    print('Starting Data Augmentation with modes:',transforms_for_data_aug)
    #for filename in tqdm(os.listdir(directory),desc='Augumented Data Prepared'):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            class_name, sample_no, frame_no = filename.split('-')
            frame_no, _ = frame_no.split('.')
            #print(int(frame_no[5:]))
            if class_name+sample_no not in files.keys():
                for t in transforms_for_data_aug:
                    files[class_name+sample_no+t] = {'label': class_name, 'frames':[0]*16}
            
            img = cv2.imread(f'{directory}/{filename}')
            img = cv2.resize(img,(112,112),interpolation = cv2.INTER_AREA)
            for t in transforms_for_data_aug:
                tempImg = transform_data_aug(img=img,transform=t)
                #cv2.imwrite(f'./AugData/{class_name}-{sample_no}-{t}-{frame_no}.jpg',tempImg)
                files[class_name+sample_no+t]['frames'][int(frame_no[5:])] = tempImg
                
    print('Data Augmentation Finished')        
            
    print(f'No_of_samples after data augmentation:{len(files)}')

    split_dict={}
    print('Starting Preprocessing for model')

    #for key in tqdm(files.keys(),desc='Preprocessed Samples'):
    for key in files.keys():
            class_name = files[key]['label']
            list_of_frames = files[key]['frames']

            #print(f'{len(list_of_frames)} for {key}')
            #print(np.where(labels==class_name)[0])
            if class_name not in split_dict:
                split_dict[class_name] = 0
            if split_dict[class_name] < 1000: #to turn val off
                train_dataset.append((pre_process_train(list_of_frames), torch.from_numpy(np.where(labels==class_name)[0])))
                split_dict[class_name] += 1    
            else:
                val_dataset.append((pre_process_train(list_of_frames), torch.from_numpy(np.where(labels==class_name)[0])))
            
    print('Preprocessing finished')
    shutil.rmtree(opt.train_directory+str(id))

    return (train_dataset,val_dataset, len(labels))

# MAIN FUNCTION
def get_fine_tuned(id):

    global opt 
    opt = parse_opts()
    
    # Opt parameters
    opt.confidence_threshold = 0.1
    opt.frame_to_send = 60
    opt.downsample = 2
    opt.width_mult = 1.0
    opt.pretrain_path = f'./PreTrained/jester_mobilenetv2_{opt.width_mult}x_RGB_16_best.pth'
    opt.n_classes = 27
    opt.sample_size = 112
    opt.scale_in_test = 1.0
    opt.no_cuda = False
    opt.model = "mobilenetv2"
    opt.arch = "mobilenetv2"
    #opt.batch_size = 8
    opt.no_train = False
    opt.n_threads = 0
    opt.ft_portion = 'last_layer'
    #opt.n_epochs = 10
    #opt.learning_rate = 0.01
    opt.use_amp = True
    
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    
    opt.train_directory = './TrainData' 
    training_data, validation_data, opt.n_finetune_classes = getData(id, opt.train_directory+str(id))
    #training_data, validation_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8)+1:]
    print(f'Received Data for {opt.n_finetune_classes} classes')
    print(f'Train Size: {len(training_data)}, Validation Size: {len(validation_data)}')
    
    # Model
    model = mobilenetv2.get_model(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                width_mult=opt.width_mult)
    
    # Model
    print('Initializing model')  
    model, parameters = intialize(model)
    print('Model Intialized')
    
    #hyperparams
    opt.n_epochs = 5
    opt.batch_size = 8
    opt.learning_rate = 0.01
    
    #report = []

    opt.top1_values = []
    opt.top3_values = []
    opt.train_loss_values = []
    opt.top1_values_val = []
    opt.top3_values_val = []
    opt.val_loss_values = []
    
    begin_time = time.time()
    print("starting training")
    model = train(model, parameters, training_data, validation_data)
    del training_data
    del validation_data
    torch.cuda.empty_cache()
    total_time = time.time() - begin_time
    print(f"Training finished in {total_time}")
    """
    df_top1 = pd.DataFrame(opt.top1_values)
    df_top1.to_csv(f'./Graph_data/top1_{lr}_{bsize}.csv',header=False,index=False)
    df_top3 = pd.DataFrame(opt.top3_values)
    df_top3.to_csv(f'./Graph_data/top3_{lr}_{bsize}.csv',header=False,index=False)
    df_train_loss = pd.DataFrame(opt.train_loss_values)
    df_train_loss.to_csv(f'./Graph_data/train_loss_{lr}_{bsize}.csv',header=False,index=False)
    
    df_top1_val = pd.DataFrame(opt.top1_values_val)
    df_top1_val.to_csv(f'./Graph_data/top1_val_{lr}_{bsize}.csv',header=False,index=False)
    df_top3_val = pd.DataFrame(opt.top3_values_val)
    df_top3_val.to_csv(f'./Graph_data/top3_val_{lr}_{bsize}.csv',header=False,index=False)
    df_val_loss = pd.DataFrame(opt.val_loss_values)
    df_val_loss.to_csv(f'./Graph_data/val_loss_{lr}_{bsize}.csv',header=False,index=False)
    
    print('lr:',lr,'batch_size:',bsize,'train_acc:',opt.top1_values[-1],'val_acc:',opt.top1_values_val[-1],'time:',total_time)

    report.append({'lr':lr,'batch_size':bsize,'train_acc':opt.top1_values,'val_acc':opt.top1_values_val,'time':total_time})
    
    print('Report:')
    print(report)
    """
    return model
    


