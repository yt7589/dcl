#coding=utf8
from __future__ import print_function, division
import os,sys,time,datetime
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
from apps.wxs.wxs_dsm import WxsDsm

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(Config, model, data_loader, val_version, epoch_num, log_file, efd=None):
    model.train(False)
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    bmy_correct = 0
    bm_correct = 0
    bb_correct = 0 # 通过bmy求出的品牌精度
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
    #
    bmy_id_bm_vo_dict = WxsDsm.get_bmy_id_bm_vo_dict()
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = Variable(data_val[0].cuda())
            print('eval_model.eval_turn inputs: {0};'.format(inputs.shape))
            brand_labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            bmy_labels = Variable(torch.from_numpy(np.array(data_val[-1])).long().cuda())
            img_files  = data_val[-2]
            outputs = model(inputs)
            loss = 0

            ce_loss = get_ce_loss(outputs[0], brand_labels).item()
            loss += ce_loss

            val_loss_recorder.update(loss)
            val_celoss_recorder.update(ce_loss)

            if Config.use_dcl and Config.cls_2xmul:
                outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
            else:
                outputs_pred = outputs[0]
            top3_val, top3_pos = torch.topk(outputs_pred, 3)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == brand_labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == brand_labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == brand_labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)
            # 求出年款精度
            outputs_bmy = outputs[-1]
            bmy_top5_val, bmy_top5_pos = torch.topk(outputs_bmy, 5)
            batch_bmy_correct = torch.sum((bmy_top5_pos[:, 0] == bmy_labels)).data.item()
            bmy_correct += batch_bmy_correct
            bb_correct = 0
            # 求出车型精度
            batch_bm_correct = 0
            for im in range(bmy_top5_pos.shape[0]):
                gt_bmy_id = bmy_top5_pos[im][0].item()
                net_bmy_id = bmy_labels[im].item()
                gt_bm_vo = bmy_id_bm_vo_dict[gt_bmy_id]
                net_bm_vo = bmy_id_bm_vo_dict[net_bmy_id]
                if gt_bm_vo['model_id'] == net_bm_vo['model_id']:
                    batch_bm_correct += 1
            bm_correct += batch_bm_correct
            # 找出品牌错误的样本，写入文件top1_error_samples
            if efd is not None:
                for idx in range(top3_pos.shape[0]):
                    if top3_pos[idx][0] != brand_labels[idx]:
                        efd.write('{0}*{1}*{2}\n'.format(
                            img_files[idx], brand_labels[idx], 
                            top3_pos[idx][0]
                        ))
            '''
            # 
            pred_size = top3_pos[:, 0].shape[0]
            batch_bb_correct = 0
            for idx in range(pred_size):
                pred_bmy = fgvc_id_brand_dict[int(top3_pos[idx][0])]
                pred_brand = pred_bmy.split('_')[0]
                gt_bmy = fgvc_id_brand_dict[int(labels[idx])]
                gt_brand = gt_bmy.split('_')[0]
                if pred_brand == gt_brand:
                    batch_bb_correct += 1
            bb_correct += batch_bb_correct
            brand_correct = 0
            '''

        val_acc1 = val_corrects1 / item_count
        val_acc2 = val_corrects2 / item_count
        val_acc3 = val_corrects3 / item_count
        bmy_acc = bmy_correct / item_count
        bm_acc = bm_correct / item_count
        bb_acc = bb_correct / item_count

        log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')


        t1 = time.time()
        since = t1-t0
        print('--'*30, flush=True)
        print('% 3d %s %s %s-loss: %.4f || 品牌：%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f; 车型:%.4f; 年款：%.4f; ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, bm_acc, bmy_acc, since), flush=True)
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

def predict_main(Config, model, data_loader, val_version, epoch_num, log_file):
    with open('./logs/top1_error_samples.txt', 'w+', encoding='utf-8') as efd:
        eval_turn(Config, model, data_loader, val_version, epoch_num, log_file, efd=efd)

def predict_main_mine(Config, model):
    print('预测图像数据...')
    correct_num = 0
    total_num = 0
    with open('./datasets/CUB_200_2011/anno/test_ds_v4.txt', 'r', encoding='utf-8') as fd:
        for line in fd:
            arrs0 = line[:-1].split('*')
            img_file = arrs0[0]
            cls_id = arrs0[1]
            predict_cls_id, conf = predict_image(Config, model, img_file)
            print('{0}: {1} ? {2};'.format(img_file, cls_id, predict_cls_id))
            if cls_id == predict_cls_id:
                correct_num += 1
            total_num += 1
    print('acc: {0};'.format(correct_num / total_num))
    '''
    imgpath1 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/中通/中通/车型02/夜#01_豫E77983_999_中通_中通_车型02_610500200970659398_0.jpg'
    cls_id, conf = predict_image(model, imgpath1)
    imgpath2 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/中通/中通/车型02/白#01_陕B33783_999_中通_中通_车型02_610500200969343138_1.jpg'
    cls_id, conf = predict_image(model, imgpath2)
    imgpath3 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/白#02_陕VAB888_006_宾利_欧陆_2007-2009_610500200969676828_0.jpg'
    cls_id, conf = predict_image(model, imgpath3)
    imgpath4 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/白#02_陕A83CN2_006_宾利_欧陆_2007-2009_610500200988158378_0.jpg'
    cls_id, conf = predict_image(model, imgpath4)
    imgpath5 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/白#02_京PR9N09_006_宾利_欧陆_2007-2009_610500200972748870_0.jpg'
    cls_id, conf = predict_image(model, imgpath5)
    imgpath6 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/白#02_陕A83CN2_006_宾利_欧陆_2007-2009_610500200988093497_0.jpg'
    cls_id, conf = predict_image(model, imgpath6)
    imgpath7 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/白#02_豫MQ0888_006_宾利_欧陆_2007-2009_610500200972940875_0.jpg'
    cls_id, conf = predict_image(model, imgpath7)
    imgpath8 = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test/宾利/欧陆/2007-2009/夜#02_陕AY02V3_006_宾利_欧陆_2007-2009_610500200973261517_0.jpg'
    cls_id, conf = predict_image(model, imgpath8)
    '''
    

def predict_image(Config, model, imgpath):
    transformers = load_data_transformers(224, 224, [3, 3])
    totensor = transformers['test_totensor']
    with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
    sample = totensor(img)
    sample = sample.view(1, sample.shape[0], sample.shape[1], sample.shape[2])
    inputs = Variable(sample.cuda())
    outputs = model(inputs)



    if Config.use_dcl and Config.cls_2xmul:
        outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
    else:
        outputs_pred = outputs[0]



    top3_val, top3_pos = torch.topk(outputs_pred, 1)
    arrs0 = imgpath.split('/')
    img = arrs0[-1]
    print('cls_id: {0}; conf: {1};  {2};'.format(top3_pos[0][0], top3_val[0][0], img))
    return top3_pos[0][0], top3_val[0][0]

