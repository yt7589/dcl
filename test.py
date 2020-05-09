#coding=utf-8
import os
import math
import sys
import datetime
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torch.onnx as onnx
from torch.onnx import OperatorExportTypes

from transforms import transforms
from utils.train_model import train
from utils.train_model import log_progress
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
import utils.dataset_DCL as dclds
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset

import pdb
import utils.utils as du

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#torch.backends.cudnn.benchmark = False

# parameters setting
def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=8, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=512, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args, flush=True)
    Config = LoadConfig(args, 'train')
    Config.cls_2 = args.cls_2
    Config.cls_2xmul = args.cls_mul
    assert Config.cls_2 ^ Config.cls_2xmul
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    val_set = dataset(Config = Config,\
                      anno = Config.val_anno,\
                      common_aug = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["test_totensor"],\
                      test=True)
    dataloader = torch.utils.data.DataLoader(val_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=True if Config.use_backbone else False,
                                                pin_memory=True)
    setattr(dataloader, 'total_item_len', len(val_set))
    setattr(dataloader, 'num_cls', Config.numcls)
    cudnn.benchmark = True
    print('Choose model and train set', flush=True)
    model = MainModel(Config)
    resume = './net_model/training_descibe_5910_CUB/weights_11_2144_0.9894_0.9980.pth'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)
    print('^_^ The End Load model is success! v0.0.1')
    model.train(False)
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = math.ceil(len(val_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            outputs = model(inputs)
            print('outputs: {0}; {1}; {2};'.format(outputs[0].shape, outputs[1].shape, outputs[2].shape))
            print('labels: {0};'.format(labels))

