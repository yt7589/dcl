#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc

import pdb, cv2
from PIL import Image
import onnx

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet18', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=4, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=224, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=224, type=int)
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--swap_num', default=[5, 5],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args


def classifier_batch(input_imgs, net):

    """
    :param img_addr: a string that contains the address to the image
    :param boundary:  optional, used for cropping the function
                should be a tuple of 4 int elements: (x_min, y_min, x_max, y_max)
     :param net: well-trained model
    :returns: a tuple (predict class, confidence score)
              predict class: from 1 to 196
    """

    crop_size = (224, 224)  # size of cropped images for 'See Better'
    


    '''
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    '''
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.225, 0.225, 0.225])
    ])
    input_ = []
    for input_path in input_imgs:
        #input_img = Image.open(input_path).convert('RGB')
        #print(input_path)
        tmp =cv2.imread(input_path)
        tmp=cv2.resize(tmp, (224,224))
        input_img = Image.fromarray(tmp)
        input_img = transform(input_img)
        input_img = torch.unsqueeze(input_img, 0)
        input_.append(input_img)
    X = torch.cat(input_, dim=0)
    X = X.to(torch.device("cuda"))


    outputs = net(X)
    #print(outputs[0].shape,outputs[1].shape,outputs[2].shape,Config.numcls,  )
    y_pred = outputs[0]# + outputs[1][:,0:Config.numcls] + outputs[1][:,Config.numcls:2*Config.numcls]

    #print(y_pred)
    if True:
        result_ = []
        for y in y_pred:
            _, pred_idx = y.topk(1, 0, True, True)

            s = torch.nn.Softmax(dim=0)
            confidence_score = max(s(y)).item()
            y_pred = pred_idx.item()
            result_.append((y_pred, confidence_score))

        return result_    

    

import shutil
def test_sample(model):

    """
    A sample function that shows how to use classifier() function
    Note: if the testing dataset annos file is not in the same format as the
    annos file provided by Stanford Cars dataset, you need to write your own
    test function

    :param annos_file: annos file address (string)
    :param predict_text: output txt file address (string)
    :return: None (output a txt file that can be verified on ImageNet server)
    """
    #net = torch.load('trained_models/resnet152_94.pkl')
    val_batch_size = 512

    pos_neg = -2
    path_imgs= getimgpath(pos_neg)
    #path_imgs = path_imgs[-1000:]
    if pos_neg < 0:
        pos_neg+=2
    #
    
    remove_imgs = []
    import random
    total = len(path_imgs)
    
    batch_imgpath = []
    
    for j, imgpath in enumerate( path_imgs):
        #if j == 8000:
            #break
        #imgpath = Image.open(imgpath)
        if j == 0:
            batch_imgpath.append(imgpath)
            if len(path_imgs) >1:
                continue
        if j %val_batch_size != 0 :
            batch_imgpath.append(imgpath)
            if j != len(path_imgs) -1 :
                continue
            
        if j %(4*val_batch_size) ==0:
            print("{}/{}".format(j,total))
        result_ = classifier_batch(batch_imgpath, model)   
        
        for imgpath, (y_pred, confidence_score) in zip(batch_imgpath, result_):
            
            if pos_neg != y_pred :
                print(imgpath, confidence_score)
                remove_imgs.append((imgpath, confidence_score))
                #out = str(confidence_score)[2:4]+'_'+os.path.basename(imgpath)
            if False and pos_neg == y_pred and confidence_score <0.7:
                print(imgpath, confidence_score)
                remove_imgs.append((imgpath, confidence_score))
                #out = str(confidence_score)[2:4]+'_'+os.path.basename(imgpath)   
        batch_imgpath = []        
    remove_imgs = list(set(remove_imgs))        
    print('total: ', len(remove_imgs))      

    out_vis_path = './tmpsave_test2'
    if os.path.exists(out_vis_path):
        shutil.rmtree(out_vis_path)
    os.makedirs(out_vis_path)    
    for i in remove_imgs:
        imgpath, confidence_score = i[0], i[1]
        if not os.path.exists(imgpath):
            print(imgpath)
            continue
        out = str(confidence_score)[2:4]+'_'+os.path.basename(imgpath)
        #os.system('mv '+imgpath + ' '+os.path.join('/home/zhangsy/tmpsave2', out))
        shutil.copy(imgpath, os.path.join(out_vis_path, out))
        #shutil.move(imgpath, os.path.join(out_vis_path))

        
        

def getimgpath(pos_neg=1):
    import sys
    
    
    
    if pos_neg ==0 :

        result_path = ['/home/ubuntu/zhangsy/lcd/direction/1']
        result_path = ['/home/ubuntu/zhangsy/lcd/testdata/train/00001']

        result_path = ['/home/ubuntu/fb/0920/back']
    elif pos_neg ==1:

        result_path = ['/home/ubuntu/zhangsy/lcd/direction/0']
        result_path = ['/home/ubuntu/zhangsy/lcd/testdata/train/00000']
    else:
        from c3ae_utils import getpath_expriment as getpath
        #from c3ae_utils import getpath_expriment as getpath
        phone_paths, none_phone_paths = getpath()
        pos_neg +=2
        
        if pos_neg ==0 :
            result_path = none_phone_paths
        elif pos_neg ==1:
            result_path = phone_paths 

    pos_imgs = []
    for p in result_path:
        files = os.listdir(p)
        tmp = [os.path.join(p, x) for x in files if os.path.splitext(x)[-1].lower() in ['.jpg', '.jpeg', '.png']]
        print(p, ': ', len(tmp))
        pos_imgs += tmp

    return pos_imgs

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    print(args)    
    Config = LoadConfig(args, args.version, True)
    cudnn.benchmark = True
    resume = './net_model/training_descibe_82121_CUB/weights_0_43999_0.9744_0.9899.pth'
    model = MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict=torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model.train(False)
    model.eval()
    run_mode = 1 # 1 dynamic; 2 static
    if 1 == run_mode:
        torch.save(model, 'dcl_v3.pth')
        dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
        torch.onnx.export(model, dummy_input, "dcl_v3_batch.onnx", verbose=True,
                         input_names=["data"], output_names=["output"], \
                            training=False, opset_version=9,
                            do_constant_folding=True,
                            dynamic_axes={"data":{0:"batch_size"},     # 批处理变量
                                    "output":{0:"batch_size"}})
        print("export finished")
        # Load the ONNX model
        model = onnx.load("dcl_v3_batch.onnx")
        # Check that the IR is well formed
        #onnx.checker.check_model(model)
        print("check_model finished")
        #onnx.helper.printable_graph(model.graph)
        exit(0)
    else:
        fixbatch =  1
        dummy_input = torch.randn(fixbatch, 3, 224, 224, device='cuda')
        torch.onnx.export(model, dummy_input, f"dcl_v3_{fixbatch}.onnx", verbose=True,
                         input_names=["data"], output_names=["output"], \
                            training=False, opset_version=9,
                            do_constant_folding=True,)
        print("export finished")
        # Load the ONNX model
        model = onnx.load(f"dcl_v3_{fixbatch}.onnx")
        # Check that the IR is well formed
        #onnx.checker.check_model(model)
        print("check_model finished")
        #onnx.helper.printable_graph(model.graph)
        exit(0)

    
    #with torch.no_grad():
    #    test_sample(model)

#python -m onnxsim dcl.onnx dlc_sim.onnx  --input-shape "1,3,224,224"
# pip install onnx-simplifier
# netron /data2/zhangsy/head/c3ae/DCL/dcl.onnx --host=172.16.20.20
# python -m onnxsim dcl.onnx dcl_sim.onnx  --input-shape "1,3,224,224"
#python -m onnxsim dcl_v3.onnx dcl_v3_sim.onnx  --input-shape "1,3,224,224"
#python -m onnxsim dcl_v3_batch.onnx dcl_v3_batch_sim.onnx  --input-shape "-1,3,224,224"
#python -m onnxsim dcl_v3_32.onnx dcl_v3_32_sim.onnx  --input-shape "32,3,224,224"
#netron ./dcl_v3_batch.onnx --host=172.16.20.20
#scp -r /home/ubuntu/zhangsy/lcd/dcl_v3_8_sim.onnx ubuntu@192.168.8.114:/home/ubuntu/ds/lzy/tiny_tensorrt_vtn/model

