#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

import PIL.Image as Image
from utils.ds_manager import DsManager
from config import load_data_transformers

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(Config, model, data_loader, val_version, epoch_num, log_file):

    model.train(False)

    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    brand_correct = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_l1_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...'%val_version, flush=True)
    _, fgvc_id_to_bmy_dict = DsManager.get_bmy_and_fgvc_id_dicts()
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            outputs = model(inputs)
            loss = 0

            ce_loss = get_ce_loss(outputs[0], labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            top3_val, top3_pos = torch.topk(outputs_pred, 3)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)
            # 求出品牌精度
            pred_size = top3_pos[:, 0].shape[0]
            batch_brand_correct = 0
            for idx in range(pred_size):
                pred_bmy = fgvc_id_to_bmy_dict['{0}'.format(top3_pos[idx][0])]
                pred_brand = pred_bmy.split('_')[0]
                gt_bmy = fgvc_id_to_bmy_dict['{0}'.format(labels[idx])]
                gt_brand = gt_bmy.split('_')[0]
                if pred_brand == gt_brand:
                    batch_brand_correct += 1
            brand_correct += batch_brand_correct

        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count
        brand_acc = brand_correct / item_count

        log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')


        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f; brand:%.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, brand_acc, since), flush=True)
        print('--' * 30, flush=True)

    return val_acc1, val_acc2, val_acc3


def filter_samples(Config, model, data_loader):
    '''
    使用模型对增量数据进行预测，将预测错误的数据记录下来
    '''
    error_samples = []
    model.train(False)
    val_corrects1 = 0
    brand_correct = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls
    _, fgvc_id_to_bmy_dict = DsManager.get_bmy_and_fgvc_id_dicts()
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            outputs = model(inputs)
            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            top3_val, top3_pos = torch.topk(outputs_pred, 3)
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            # 求出品牌精度
            pred_size = top3_pos[:, 0].shape[0]
            batch_brand_correct = 0
            for idx in range(pred_size):
                pred_bmy = fgvc_id_to_bmy_dict['{0}'.format(top3_pos[idx][0])]
                pred_brand = pred_bmy.split('-')[0]
                gt_bmy = fgvc_id_to_bmy_dict['{0}'.format(labels[idx])]
                gt_brand = gt_bmy.split('-')[0]
                if pred_brand == gt_brand:
                    batch_brand_correct += 1
                if top3_pos[idx][0] != labels[idx]:
                    error_samples.append('{0}:{1}=>{2};'.format(data_val[-1][idx], pred_bmy, gt_bmy))
            brand_correct += batch_brand_correct
        val_acc1 = val_corrects1 / item_count
        brand_acc = brand_correct / item_count
        t1 = time.time()
        since = t1-t0
        print('top1: {0}; brand: {1};'.format(val_acc1, brand_acc))
        with open('./logs/error_samples.txt', 'w+', encoding='utf-8') as fd:
            print('共有错误样本{0}个，如下所示：'.format(len(error_samples)))
            for es in error_samples:
                fd.write('{0}\n'.format(es))
    return val_acc1

def predict_image(model):
    print('预测图像数据...')
    transformers = load_data_transformers(224, 224, [3, 3])
    totensor = transformers['test_totensor']
    imgpath = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/中通/中通/车型02/夜#01_豫E77983_999_中通_中通_车型02_610500200970659398_0.jpg'
    with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    sample = totensor(img)
    Variable(sample.cuda())
    outputs = model(inputs)
    outputs_pred = outputs[0]
    top3_val, top3_pos = torch.topk(outputs_pred, 3)
    print('^_^')

