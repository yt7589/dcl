#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

import pdb
from models.LoadModel import MainModel

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation
    brand_weight = 1.5 # 决定品牌分支在学习中权重
    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    steps = np.array([], dtype=np.int)
    train_accs = np.array([], dtype=np.float32)
    test_accs = np.array([], dtype=np.float32)
    ce_losses = np.array([], dtype=np.float32)
    ce_loss_mu = -1
    ce_loss_std = 0.0

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    for epoch in range(start_epoch,epoch_num-1):
        model.train(True)
        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names, brand_labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                brand_labels = Variable(torch.from_numpy(np.array(brand_labels)).cuda())

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names, brand_labels = data
                org_labels = labels
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                brand_labels = Variable(torch.from_numpy(np.array(brand_labels)).cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            optimizer.zero_grad()

            if inputs.size(0) < 2*train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)

            if Config.use_focal_loss:
                ce_loss_bmy = get_focal_loss(outputs[0], labels)
                #ce_loss_brand = get_focal_loss(outputs[-1], brand_labels)
            else:
                ce_loss_bmy = get_ce_loss(outputs[0], labels)
                #ce_loss_brand = get_ce_loss(outputs[-1], brand_labels)
            ce_loss = ce_loss_bmy # + brand_weight * ce_loss_brand

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss
            ce_loss_val = ce_loss.detach().item()
            ce_losses = np.append(ce_losses, ce_loss_val)

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law) * gamma_
                loss += law_loss

            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            exp_lr_scheduler.step(epoch)
            torch.cuda.synchronize()

            if Config.use_dcl:
                if ce_loss_mu > 0 and ce_loss_val > ce_loss_mu + 3.0*ce_loss_std:
                    # 记录下这个批次，可能是该批次有标注错误情况
                    print('记录可疑批次信息: loss={0}; threshold={1};'.format(ce_loss_val, ce_loss_mu + 2.0*ce_loss_std))
                    with open('./logs/abnormal_samples_{0}_{1}_{2}.txt'.format(epoch, step, ce_loss_val), 'a+') as fd:
                        error_batch_len = len(img_names)
                        for i in range(error_batch_len):
                            fd.write('{0} <=> {1};\r\n'.format(org_labels[i*2], img_names[i]))
                print('epoch{}: step: {:-8d} / {:d} loss=ce_loss+'
                            'swap_loss+law_loss: {:6.4f} = {:6.4f} '
                            '+ {:6.4f} + {:6.4f} '.format(
                                epoch, step % train_epoch_step, 
                                train_epoch_step, 
                                loss.detach().item(), 
                                ce_loss_val, 
                                swap_loss.detach().item(), 
                                law_loss.detach().item()), flush=True
                            )
                
            if Config.use_backbone:
                print('epoch{}: step: {:-8d} / {:d} loss=ce_loss+'
                            'swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(
                                epoch, step % train_epoch_step, 
                                train_epoch_step, 
                                loss.detach().item(), 
                                ce_loss.detach().item()), flush=True
                            )
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                '''
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False
                '''
                print('##### validate dataset #####')
                trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(
                    Config, model, data_loader['val'], 'val', epoch, log_file
                ) #eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                print('##### test dataset #####')
                val_acc1, val_acc2, val_acc3 = trainval_acc1, trainval_acc2, \
                            trainval_acc3 # eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)
                steps = np.append(steps, step)
                train_accs = np.append(train_accs, trainval_acc1)
                test_accs = np.append(test_accs, val_acc1)

                save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                torch.cuda.empty_cache()
                # 保存精度等信息并初始化
                ce_loss_mu = ce_losses.mean()
                ce_loss_std = ce_losses.std()
                print('Cross entropy loss: mu={0}; std={1}; range:{2}~{3};'.format(
                    ce_loss_mu, ce_loss_std, 
                    ce_loss_mu - 3.0*ce_loss_std,
                    ce_loss_mu + 3.0 * ce_loss_std
                ))
                ce_losses = np.array([], dtype=np.float32)
                if train_accs.shape[0] > 30:
                    np.savetxt('./logs/steps1.txt', (steps,))
                    np.savetxt('./logs/train_accs1.txt', (train_accs,))
                    np.savetxt('./logs/test_accs1.txt', (test_accs,))
                    steps = np.array([], dtype=np.int)
                    train_accs = np.array([], dtype=np.float32)
                    test_accs = np.array([], dtype=np.float32)
                

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()


    log_file.close()

def log_progress(step, train_acc, test_acc):
    # 以添加形式保存step
    with open('./logs/step.txt', 'a+') as step_fd:
        step_fd.write('{0:d},'.format(step))
    # 以添加形式保存train_acc
    with open('./logs/train_acc.txt', 'a+') as train_acc_fd:
        train_acc_fd.write('{0:.4f},'.format(train_acc))
    # 以添加形式保存test_acc
    with open('./logs/test_acc.txt', 'a+') as test_acc_fd:
        test_acc_fd.write('{0:.4f},'.format(test_acc))

def prepare_cluster_data(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    '''
    为图像聚类分析准备特征值，取DCL最后一层2048维的向量作为特征，然后再用DBScan和t-SNE进行降维处理
    '''
    model.train(False)
    with torch.no_grad():
        with open('./logs/cluster_imgs.txt', 'w+', encoding='utf-8') as imgs_fd:
            for batch_cnt_val, data_val in enumerate(data_loader['train']):
                imgs = data_val[4]
                inputs = Variable(data_val[0].cuda())
                labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
                features = model(inputs, run_mode=MainModel.RUN_MODE_FEATURE_EXTRACT)
                for img in imgs:
                    imgs_fd.write('{0}\n'.format(img))
                print('f1: {0};'.format(features.shape))
                features = features.view(-1, 2048)[:30, :]
                print('f2: {0}; {1};'.format(features.shape, type(features)))
                features = features.detach().cpu().numpy()
                np.savetxt('./logs/cluster_features.txt', features)





