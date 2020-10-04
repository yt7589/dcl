# 数据集管理类，负责生成数据描述文件
import os
from os import stat
import sys
import time
import json
import shutil
import random
import datetime
import pickle
import requests
from multiprocessing import Queue
import threading
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from torch import full
from apps.wxs.controller.c_brand import CBrand
from apps.wxs.controller.c_model import CModel
from apps.wxs.controller.c_bmy import CBmy
from apps.wxs.controller.c_sample import CSample
from apps.wxs.controller.c_dataset import CDataset
import PIL.Image as Image
from PIL import ImageStat
from apps.wxs.vdc.vd_json_saver import VdJsonSaver
from apps.wxs.vdc.vd_json_manager import VdJsonManager
#
from apps.wxs.bid.bid_vin import BidVin
from apps.wxs.dm.gcc2n_dm import Gcc2nDm
from datasets.image_lmdb import ImageLmdb
from apps.wxs.fu.file_util import FileUtil

class WxsDsm(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsDsm'

    @staticmethod
    def know_init_status():
        '''
        统计当前信息，主要是所里标书中的品牌车型年款和我们的品牌车型
        年款之间的差异
        '''
        bid_brand_set, bid_model_set, bid_bmy_set, bid_vin_set, _, _, _, _ = WxsDsm._get_bid_info()
        print('标书要求：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(bid_vin_set), len(bid_brand_set), len(bid_bmy_set)
        ))
        our_brand_set, our_model_set, our_bmy_set, our_vin_set, _ = WxsDsm._get_our_info()
        print('自有情况：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(our_vin_set), len(our_brand_set), len(our_bmy_set)
        ))
        print('******************************************************')
        # 统计品牌情况
        common_brand_set = our_brand_set & bid_brand_set
        print('我们和标书公共品牌：{0}个'.format(len(common_brand_set)))
        oh_brand_set = our_brand_set - bid_brand_set
        print('我们有标书没有品牌：{0}个；'.format(len(oh_brand_set)))
        WxsDsm.write_set_to_file(oh_brand_set, './logs/we_had_brand.txt')
        bh_brand_set = bid_brand_set - our_brand_set
        print('标书有我们没有的品牌：{0}个'.format(len(bh_brand_set)))
        WxsDsm.write_set_to_file(bh_brand_set, './logs/bid_had_brand.txt')
        all_brand_set = our_brand_set | bid_brand_set
        print('共有品牌：{0}个'.format(len(all_brand_set)))
        print('******************************************************')
        # 统计年款情况
        common_bmy_set = our_bmy_set & bid_bmy_set
        print('我们和标书公共年款：{0}个'.format(len(common_bmy_set)))
        oh_bmy_set = our_bmy_set - bid_bmy_set
        print('我们有标书没有年款：{0}个'.format(len(oh_bmy_set)))
        WxsDsm.write_set_to_file(oh_bmy_set, './logs/we_had_bmy.txt')
        bh_bmy_set = bid_bmy_set - our_bmy_set
        print('标书有我们没有年款：{0}个'.format(len(bh_bmy_set)))
        WxsDsm.write_set_to_file(bh_bmy_set, './logs/bid_had_bmy.txt')
        all_bmy_set = our_bmy_set | bid_bmy_set
        print('共有年款：{0}个'.format(len(all_bmy_set)))
        print('******************************************************')
        # 统计车辆识别码情况
        common_vin_set = our_vin_set & bid_vin_set
        print('我们和标书共有车辆识别码：{0}个'.format(len(common_vin_set)))
        oh_vin_set = our_vin_set - bid_vin_set
        print('我们有标书没有车辆识别码：{0}个'.format(len(oh_vin_set)))
        WxsDsm.write_set_to_file(oh_vin_set, './logs/we_had_vin.txt')
        bh_vin_set = bid_vin_set - our_vin_set
        print('标书有我们没有车辆识别码：{0}个'.format(len(bh_vin_set)))
        WxsDsm.write_set_to_file(bh_vin_set, './logs/bid_had_vin.txt')
        all_vin_set = our_vin_set | bid_vin_set
        print('共有车辆识别码：{0}个'.format(len(all_vin_set)))
    @staticmethod
    def _get_our_info():
        print('掌握当前情况')
        brand_set = set()
        model_set = set()
        bmy_set = set()
        vin_set = set()
        vin_bmy_dict = {}
        with open('./work/ggh_to_bmy_dict.txt', 'r', encoding='utf-8') as gfd:
            for line in gfd:
                row = line.strip()
                arrs0 = row.split(':')
                vin_code = arrs0[0]
                vin_set.add(vin_code)
                arrs1 = arrs0[1].split('-')
                brand_name = '{0}牌'.format(arrs1[0])
                brand_set.add(brand_name)
                model_name_postfix = arrs1[1]
                model_name = '{0}-{1}'.format(brand_name, model_name_postfix)
                model_set.add(model_name)
                year_name = arrs1[2]
                bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name_postfix, year_name)
                bmy_set.add(bmy_name)
                vin_bmy_dict[vin_code] = {
                    'bmy_name': bmy_name,
                    'is_imported_vehicle': 0
                }
        return brand_set, model_set, bmy_set, vin_set, vin_bmy_dict
    @staticmethod
    def _get_bid_info():
        brand_set = set()
        brand_code_dict = {}
        model_set = set()
        model_code_dict = {}
        bmy_set = set()
        bmy_code_dict = {}
        vin_set = set()
        vin_bmy_dict = {}
        seq = 0
        with open('./logs/bid_20200708.csv', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                row = line.strip()
                arrs0 = row.split(',')
                if seq > 0:
                    brand_name = arrs0[2]
                    brand_set.add(brand_name)
                    brand_code_dict[brand_name] = arrs0[1]
                    model_name_postfix = arrs0[4]
                    model_name = '{0}-{1}'.format(brand_name, model_name_postfix)
                    model_set.add(model_name)
                    model_code_dict[model_name] = arrs0[3]
                    year_name = arrs0[6]
                    bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name_postfix, year_name)
                    bmy_set.add(bmy_name)
                    bmy_code_dict[bmy_name] = arrs0[5]
                    vin_code = arrs0[8]
                    vin_set.add(vin_code)
                    vin_bmy_dict[vin_code] = {
                        'bmy_name': bmy_name,
                        'bmy_code': arrs0[5],
                        'is_imported_vehicle': arrs0[7]
                    }
                seq += 1
        return brand_set, model_set, bmy_set, vin_set, brand_code_dict, model_code_dict, bmy_code_dict, vin_bmy_dict

    @staticmethod
    def write_set_to_file(set_obj, filename):
        ''' 将集合内容写到文件中，写入前先对集合内容进行排序 '''
        lst = list(set_obj)
        lst.sort()
        with open(filename, 'w+', encoding='utf-8') as wfd:
            for item in lst:
                wfd.write('{0}\n'.format(item))
    
    @staticmethod
    def initialize_db():
        bid_brand_set, bid_model_set, bid_bmy_set, bid_vin_set, \
                    brand_code_dict, model_code_dict, \
                    bmy_code_dict, bid_vin_bmy_dict = WxsDsm._get_bid_info()
        our_brand_set, our_model_set, our_bmy_set, our_vin_set, \
                    our_vin_bmy_dict = WxsDsm._get_our_info()
        # 保存品牌信息
        brands = our_brand_set | bid_brand_set
        WxsDsm.store_brands_to_db(brands, brand_code_dict)
        # 保存车型信息
        models = our_model_set | bid_model_set
        WxsDsm.store_models_to_db(models, model_code_dict)
        # 保存年款和车辆识别码信息
        vins = our_vin_set | bid_vin_set
        WxsDsm.store_vin_bmy_to_db(vins, bid_vin_bmy_dict, our_vin_bmy_dict)


    @staticmethod
    def store_brands_to_db(brands, brand_code_dict):
        brands = list(brands)
        brands.sort()
        num = 1
        for brand_name in brands:
            if brand_name in brand_code_dict:
                source_type = 1
                brand_code = brand_code_dict[brand_name]
            else:
                source_type = 2
                brand_code = 'x{0:04d}'.format(num)
            CBrand.add_brand(brand_name, brand_code, source_type)
            num += 1

    @staticmethod
    def store_models_to_db(models, model_code_dict):
        models = list(models)
        models.sort()
        num = 1
        for model_name in models:
            arrs0 = model_name.split('-')
            brand_name = arrs0[0]
            model_name_postfix = arrs0[1]
            brand_vo = CBrand.get_brand_by_name(brand_name)
            if model_name in model_code_dict:
                model_code = model_code_dict[model_name]
                source_type = 1
            else:
                model_code = 'x{0:05d}'.format(num)
                source_type = 2
            num += 1
            CModel.add_model(model_name, model_code, brand_vo, source_type)

    SOURCE_TYPE_BID = 1
    SOURCE_TYPE_OUR = 2
    @staticmethod
    def store_vin_bmy_to_db(vins, bid_vin_bmy_dict, our_vin_bmy_dict):
        print('处理年款和车辆识别码 ^_^')
        vins = list(vins)
        vins.sort()
        num = 1
        for vin in vins:
            print('处理车辆识别码：{0}...'.format(vin))
            if vin in bid_vin_bmy_dict:
                bmy_obj = bid_vin_bmy_dict[vin]
                bmy_id = WxsDsm._process_bmy(bmy_obj['bmy_name'], bmy_obj['bmy_code'], bmy_obj['is_imported_vehicle'])
                WxsDsm._process_vin(vin, bmy_id, WxsDsm.SOURCE_TYPE_BID)
            elif vin in our_vin_bmy_dict:
                bmy_obj = our_vin_bmy_dict[vin]
                bmy_id = WxsDsm._process_bmy(bmy_obj['bmy_name'], 'b{0:05d}'.format(num), bmy_obj['is_imported_vehicle'])
                WxsDsm._process_vin(vin, bmy_id, WxsDsm.SOURCE_TYPE_OUR)
            else:
                print('异常vin：{0};'.format(vin))
            num += 1

    @staticmethod
    def _process_bmy(bmy_name, bmy_code, is_imported_vehicle):
        # 求出brand_id和brand_code
        # 求出model_id和model_code
        # 将bmy保存到t_bmy中并获取bmy_id（重复的bmy不重复加入）
        # 将vin和bmy_id保存到t_vin表中
        arrs0 = bmy_name.split('-')
        brand_name = arrs0[0]
        brand_vo = CBrand.get_brand_by_name(brand_name)
        if brand_vo is None:
            print('找不到品牌：{0};'.format(brand_name))
            sys.exit(0)
        model_name = '{0}-{1}'.format(arrs0[0], arrs0[1])
        model_vo = CModel.get_model_by_name(model_name)
        if model_vo is None:
            print('找不到车型：{0}；'.format(model_name))
        bmy_id = CBmy.add_bmy(bmy_name, bmy_code, 
            brand_vo['brand_id'], brand_vo['brand_code'],
            model_vo['model_id'], model_vo['model_code']
        )
        return bmy_id

    @staticmethod
    def _process_vin(vin_code, bmy_id, source_type):
        CBmy.add_vin(vin_code, bmy_id, source_type)

    @staticmethod
    def generate_samples():
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        oprr_num = 0
        with open('./logs/conflicts.txt', 'w+', encoding='utf-8') \
                        as WxsDsm.g_cfd:
            with open('./support/samples_v001.txt', 'w+', encoding='utf-8') \
                            as sfd:
                with open('./support/error_vins.txt', 'w+', encoding='utf-8') \
                                as efd:
                    # 进口车目录
                    folder_name = '/media/zjkj/work/fgvc_dataset/raw'
                    base_path = Path(folder_name)
                    oprr_num = WxsDsm.generate_imported_vehicle_samples(
                                oprr_num, vin_code_bmy_id_dict, 
                                bmy_id_bmy_name_dict, 
                                brand_set, base_path, sfd, efd)
                    # 国产车已处理
                    domestic1_path = Path('/media/zjkj/work/'\
                                'guochanchezuowan-all')
                    oprr_num = WxsDsm.generate_domestic_vehicle_samples(
                                oprr_num,vin_code_bmy_id_dict,  
                                bmy_id_bmy_name_dict, 
                                brand_set, 
                                domestic1_path, sfd, efd)
        print('已经处理品牌数：{0};'.format(len(brand_set)))

    @staticmethod
    def generate_imported_vehicle_samples(oprr_num, vin_code_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, base_path, sfd, efd):
        brand_num = 0
        for brand_obj in base_path.iterdir():
            brand_num += 1
            for model_obj in brand_obj.iterdir():
                for year_obj in model_obj.iterdir():
                    for sub_obj in year_obj.iterdir():
                        filename = str(sub_obj)
                        item_name = filename.split('/')[-1]
                        if not sub_obj.is_dir() and filename.endswith(
                                    ('jpg','png','jpeg','bmp')) and not \
                                        item_name.startswith('白') \
                                        and not item_name.startswith('夜'): 
                                        # 忽略其下目录
                            oprr_num = WxsDsm.process_one_img_file(
                                        oprr_num, vin_code_bmy_id_dict, 
                                        bmy_id_bmy_name_dict, brand_set, 
                                        sub_obj, sfd, efd)
        return oprr_num


    @staticmethod
    def generate_domestic_vehicle_samples(oprr_num, vin_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, 
                path_obj, sfd, efd):
        for branch_obj in path_obj.iterdir():
            for vin_obj in branch_obj.iterdir():
                for file_obj in vin_obj.iterdir():
                    filename = str(file_obj)
                    if not file_obj.is_dir() and filename.endswith(
                                    ('jpg','png','jpeg','bmp')): 
                                    # 忽略其下目录
                        oprr_num = WxsDsm.process_one_img_file(
                                    oprr_num, vin_bmy_id_dict, 
                                    bmy_id_bmy_name_dict, brand_set, 
                                    file_obj, sfd, efd)
        return oprr_num

    @staticmethod
    def process_one_img_file(oprr_num, vin_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, sub_obj, sfd, efd):
        sub_file = str(sub_obj)
        #print('处理文件：{0};'.format(sub_obj))
        arrs0 = sub_file.split('/')
        filename = arrs0[-1]
        arrs1 = filename.split('_')
        raw_vin_code = arrs1[0]
        arrs2 = raw_vin_code.split('#')
        vin_code = arrs2[0]
        if vin_code in vin_bmy_id_dict:
            bmy_id = vin_bmy_id_dict[vin_code]
        else:
            vin_had_bmy_id = False
            for k, _ in vin_bmy_id_dict.items():
                if k.startswith(vin_code):
                    bmy_id = vin_bmy_id_dict[k]
                    vin_had_bmy_id = True
                    break
            if not vin_had_bmy_id:
                bmy_id = -1
                if vin_code != '白' and vin_code != '夜':
                    efd.write('{0}\n'.format(vin_code))
        if bmy_id > 0:
            sfd.write('{0}*{1}\n'.format(sub_file, bmy_id - 1))
            bmy_name = bmy_id_bmy_name_dict[bmy_id]
            arrsn = bmy_name.split('-')
            brand_name = arrsn[0]
            brand_set.add(brand_name)
        oprr_num += 1
        if oprr_num % 1000 == 0:
            print('处理{0}条记录...'.format(oprr_num))
        return oprr_num

    
    g_bmy_id_bmy_name_dict = None 
    g_vin_bmy_id_dict = None
    g_brand_set = None
    g_error_num = 0
    g_cfd = None
    @staticmethod
    def generate_samples_org():
        vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        WxsDsm.g_bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        WxsDsm.g_vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        WxsDsm.g_brand_set = set()
        with open('./logs/conflicts.txt', 'w+', encoding='utf-8') as WxsDsm.g_cfd:
            with open('./support/samples.txt', 'w+', encoding='utf-8') as sfd:
                with open('./support/error_vins.txt', 'w+', encoding='utf-8') as efd:
                    # 进口车目录
                    folder_name = '/media/zjkj/work/fgvc_dataset/raw'
                    base_path = Path(folder_name)
                    WxsDsm.generate_samples_from_path(vin_bmy_id_dict, base_path, sfd, efd)
                    # 国产车已处理
                    domestic1_path = Path('/media/zjkj/work/guochanchezuowan-all')
                    WxsDsm.generate_samples_from_path_domestic(vin_bmy_id_dict, domestic1_path, sfd, efd)
        print('已经处理品牌数：{0};'.format(len(WxsDsm.g_brand_set)))

    opr_num = 1
    err_num = 0
    g_dif = 0
    @staticmethod
    def generate_samples_from_path(vin_bmy_id_dict, path_obj, sfd, efd):
        #with open('./logs/samples.txt', 'w+', encoding='utf-8') as sfd:
        brand_num = 0
        for brand_obj in path_obj.iterdir():
            brand_num += 1
            for model_obj in brand_obj.iterdir():
                for year_obj in model_obj.iterdir():
                    for sub_obj in year_obj.iterdir():
                        filename = str(sub_obj)
                        if not sub_obj.is_dir() and filename.endswith(
                                    ('jpg','png','jpeg','bmp')): # 忽略其下目录
                            WxsDsm.process_one_img_file(vin_bmy_id_dict, sub_obj, sfd, efd)
                            item_name = filename.split('/')[-1]
                            if (WxsDsm.g_dif != brand_num - len(WxsDsm.g_brand_set)) and not item_name.startswith('白') \
                                        and not item_name.startswith('夜'):
                                WxsDsm.g_dif = brand_num - len(WxsDsm.g_brand_set)
                                arrs0 = item_name.split('_')
                                arrs1 = arrs0[0].split('#')
                                vin_code = arrs1[0]
                                bmy_id = WxsDsm.g_vin_bmy_id_dict[vin_code]
                                #bmy_id = CBmy.get_bmy_id_by_vin_code(vin_code)[0]
                                #bmy_vo = CBmy.get_bmy_by_id(bmy_id)
                                bmy_name = WxsDsm.g_bmy_id_bmy_name_dict[bmy_id]
                                WxsDsm.g_cfd.write('我们：{0} <=> 标书：{1}；目录品牌数：{2}；汇总品牌数：{3}\n'.format(filename, bmy_name, brand_num, len(WxsDsm.g_brand_set)))
                                WxsDsm.err_num += 1

    @staticmethod
    def generate_samples_from_path_domestic(vin_bmy_id_dict, path_obj, sfd, efd):
        for branch_obj in path_obj.iterdir():
            for vin_obj in branch_obj.iterdir():
                for file_obj in vin_obj.iterdir():
                    filename = str(file_obj)
                    if not file_obj.is_dir() and filename.endswith(
                                    ('jpg','png','jpeg','bmp')): # 忽略其下目录
                        WxsDsm.process_one_img_file(vin_bmy_id_dict, file_obj, sfd, efd)

    @staticmethod
    def generate_dataset():
        print('生成数据集...')
        #vins = CBmy.get_vin_codes()
        vins = CBmy.get_wxs_vins()
        vin_samples_dict = WxsDsm.get_vin_samples_dict()
        with open('./support/raw_bid_train_ds.txt', 'w+', encoding='utf-8') as train_fd:
            with open('./support/raw_bid_test_ds.txt', 'w+', encoding='utf-8') as test_fd:
                for vin in vins:
                    print('处理：{0} <=> {1};'.format(vin['vin_id'], vin['vin_code']))
                    if vin['vin_code'] in vin_samples_dict:
                        samples = vin_samples_dict[vin['vin_code']]
                        samples_num = len(samples)
                        if samples_num >= 1000:
                            WxsDsm.process_bt_1000_samples(samples, train_fd, test_fd)
                        elif samples_num >= 100 and samples_num < 1000:
                            WxsDsm.process_100_to_1000_samples(samples, train_fd, test_fd)
                        elif samples_num >= 10 and samples_num < 100:
                            WxsDsm.process_10_to_100_samples(samples, train_fd, test_fd)
                        elif samples_num >= 1 and samples_num < 10:
                            WxsDsm.process_lt_10_samples(samples, train_fd, test_fd)
                        else:
                            print('该车辆识别码{0}没有样本记录...'.format(vin['vin_code']))

    @staticmethod
    def get_vin_samples_dict():
        '''
        获得一个字典，key值为车辆识别码，值为该车辆识别下样本列表
        '''
        bmy_id_vin_dict = CBmy.get_bmy_id_vin_dict()
        vin_samples_dict = {}
        samples = []
        with open('./support/samples.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs = line.split('*')
                bmy_id = int(arrs[1]) + 1
                if bmy_id in bmy_id_vin_dict:
                    vin_code = bmy_id_vin_dict[bmy_id]
                    if vin_code not in vin_samples_dict:
                        vin_samples_dict[vin_code] = [{'img_file': arrs[0], 'bmy_id': bmy_id -1}]
                    else:
                        vin_samples_dict[vin_code].append({'img_file': arrs[0], 'bmy_id': bmy_id -1})
        return vin_samples_dict

    @staticmethod
    def process_bt_1000_samples(samples, train_fd, test_fd):
        '''
        随机则取10张作为测试数据集，其余作为训练数据集
        '''
        data = list(range(len(samples)))
        random.shuffle(data)
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@1 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
        train_idxs = data[10:1011]
        print('训练数据集：')
        for idx in train_idxs:
            print('#1 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))

    @staticmethod
    def process_100_to_1000_samples(samples, train_fd, test_fd):
        '''
        随机则取10张作为测试数据集，其余作为训练数据集
        '''
        data = list(range(len(samples)))
        random.shuffle(data)
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@2 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
        train_idxs = data[10:]
        print('训练数据集：')
        for idx in train_idxs:
            print('#2 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))

    @staticmethod
    def process_10_to_100_samples(samples, train_fd, test_fd):
        '''
        随机取10张作为测试数据集，取全部图片作为训练数据集
        '''
        data = list(range(len(samples)))
        random.shuffle(data)
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@3 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
        train_idxs = data
        print('训练数据集：')
        for idx in train_idxs:
            print('#3 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))

    @staticmethod
    def process_lt_10_samples(samples, train_fd, test_fd):
        data = list(range(len(samples)))
        random.shuffle(data)
        test_idxs = data
        print('测试数据集：')
        for idx in test_idxs:
            print('@4 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
        train_idxs = data
        print('训练数据集：')
        for idx in train_idxs:
            print('#4 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])))

    @staticmethod
    def report_current_status():
        '''
        列出数据集中品牌数和品牌列表，年款数和年款列表，当前未覆盖
        品牌数和品牌列表，未覆盖年款数和年款列表
        '''
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        bmy_set = set()
        with open('./logs/samples.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[1]) - 1
                bmy_name = bmy_id_bmy_name_dict[bmy_id]
                bmy_set.add(bmy_name)
                arrs1 = bmy_name.split('-')
                brand_name = arrs1[0]
                print('已有品牌{1}：{0};'.format(brand_name, arrs0[1]))
                brand_set.add(brand_name)
        print('当前覆盖品牌数：{0}; 覆盖年款数：{1};'.format(len(brand_set), len(bmy_set)))
        with open('./logs/had_brands.txt', 'w+', encoding='utf-8') as wfd:
            for brand_name in brand_set:
                wfd.write('{0}\n'.format(brand_name))
        with open('./logs/had_bmys.txt', 'w+', encoding='utf-8') as yfd:
            for bmy_name in bmy_set:
                yfd.write('{0}\n'.format(bmy_name))

    @staticmethod
    def compare_our_brands_and_bid_brands():
        vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        WxsDsm.g_bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        WxsDsm.g_vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        wb_brand_dict = {}
        we_had_bid_no = set()
        done_brand = set()
        with open('./logs/wb_brand.txt', 'r', encoding='utf-8') as wb_fd:
            for line in wb_fd:
                line = line.strip()
                we_had_bid_no.add(line)
        for brand_name in we_had_bid_no:
            base_path = Path('/media/zjkj/work/fgvc_dataset/raw/{0}'.format(brand_name[:-1]))
            is_break = False
            print('品牌名称：{0}:'.format(brand_name))
            for model_obj in base_path.iterdir():
                for year_obj in model_obj.iterdir():
                    for item_obj in year_obj.iterdir():
                        item_name = str(item_obj)
                        if not item_obj.is_dir() and item_name.endswith(
                                    ('jpg','png','jpeg','bmp')): # 忽略其下目录
                            arrs0 = item_name.split('/')
                            arrs1 = arrs0[-1].split('_')
                            arrs2 = arrs1[0].split('#')
                            vin_code = arrs2[0]
                            if vin_code in vin_bmy_id_dict:
                                bmy_id = vin_bmy_id_dict[vin_code]
                            elif vin_code[:8] in vin_bmy_id_dict:
                                bmy_id = vin_bmy_id_dict[vin_code[:8]]
                            else:
                                #wfd.write('############## {0}\n'.format(vin_code))
                                bmy_id = -1
                            print('正在处理：{0}; vin_code={1}; bmy_id={2};'.format(item_name, vin_code, bmy_id))
                            if bmy_id > 0:
                                print('      正常品牌：bmy_id={0};'.format(bmy_id))
                                bmy_name = WxsDsm.g_bmy_id_bmy_name_dict[bmy_id]
                                arrsn = bmy_name.split('-')
                                brand_name1 = arrsn[0]
                                if brand_name not in wb_brand_dict:
                                    wb_brand_dict[brand_name] = brand_name1
                                    done_brand.add(brand_name)
                                is_break = True
                                break # break item
                    if is_break:
                        break # break year
                if is_break:
                    break # break model
        for k, v in wb_brand_dict.items():
            print('### {0}: {1};'.format(k, v))
        print('共有{0}个；'.format(len(wb_brand_dict)))
        diff_set = we_had_bid_no - done_brand
        print('未找到匹配关系的品牌：{0}个；如下所示：'.format(len(diff_set)))
        for bn in diff_set:
            print('### {0};'.format(bn))

    @staticmethod
    def find_bad_images(sub_dir):
        '''
        检查某个子目录下图片是否有破损情况
        '''
        bad_files = []
        base_path = Path('/media/zjkj/work/guochanchezuowan-all/{0}'.format(sub_dir))
        num = 0
        for vph_obj in base_path.iterdir():
            for item_obj in vph_obj.iterdir():
                item_name = str(item_obj)
                if not item_obj.is_dir() and item_name.endswith(
                                ('jpg','png','jpeg','bmp')): # 忽略其下目录
                    img_path = str(item_obj)
                    try:
                        with open(img_path, 'rb') as f:
                            with Image.open(f) as img:
                                img.convert('RGB')
                        num += 1
                        if num % 100 == 0:
                            print('已经处理图片：{0};'.format(num))
                    except OSError as ex:
                        print('{0}: {1};'.format(img_path, ex))
                        bad_files.append(img_path)
        with open('./logs/bad_images.txt', 'w+', encoding='utf-8') as bfd:
            for img in bad_files:
                bfd.write('{0}\n'.format(img))

    @staticmethod
    def process_g2_folder_main():
        dst_base_folder = '/media/zjkj/work/guochanche_2n'
        with open('./logs/g2.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                WxsDsm.process_g2_folder(line, dst_base_folder)

    @staticmethod
    def process_g2_folder(folder, dst_base_folder):
        '''
        处理guochanche_2目录下一个最底层目录，将其中图片文件拷贝到
        guochanche_2n目录下，以车辆识别码为目录名，将同样车辆识
        别码的图片放到该目录下
        参数：
            folder：guochanche_2下面最底层目录，字符串类型
            dst_folder：guochanche_2n目录，字符串
        '''
        base_path = Path(folder)
        for item_obj in base_path.iterdir():
            item_str = str(item_obj)
            if item_str.endswith(('jpg','png','jpeg','bmp')):
                print('处理图形文件：{0}...'.format(item_str))
                arrs0 = item_str.split('/')
                filename = arrs0[-1]
                arrs1 = arrs0[-1].split('_')
                arrs2 = arrs1[0].split('#')
                vin_code = arrs2[0]
                dst_folder = '{0}/{1}'.format(dst_base_folder, vin_code)
                if not os.path.exists(dst_folder):
                    os.mkdir(dst_folder)
                dst_file = '{0}/{1}'.format(dst_folder, filename)
                shutil.move(item_str, dst_file)
                        

    @staticmethod
    def get_leaf_folders_main():
        '''
        求出guochanche_2下所有最底层子目录并打印
        '''
        src_dir = '/media/zjkj/work/guochanche_2'
        src_path = Path(src_dir)
        leaf_folder_set = set()
        WxsDsm.get_leaf_folders(src_path, leaf_folder_set)
        print('共有{0}个文件；'.format(WxsDsm.total_files))
        for folder in leaf_folder_set:
            print(folder)

    total_files = 0
    @staticmethod
    def get_leaf_folders(base_path, leaf_folder_set):
        '''
        采用递归方式求出guochanche_2下所有最底层子目录，添加到
        leaf_foler_set集合中
        参数：
            base_path：父目录Path对象
            leaf_folder_set：底层目录集合
        '''
        if not base_path.is_dir():
            return
        # 列出所有最子一级目录
        for item_obj in base_path.iterdir():
            item_str = str(item_obj)
            if item_obj.is_dir():
                print('parent: {0}；当前文件数：{2};\n    child: {1}'.format(item_obj.parent, item_str, WxsDsm.total_files))
                if item_obj.parent in leaf_folder_set:
                    leaf_folder_set.remove(str(item_obj.parent))
                leaf_folder_set.add(item_str)
                WxsDsm.get_leaf_folders(item_obj, leaf_folder_set)
            else:
                if item_str.endswith(('jpg','png','jpeg','bmp')):
                    WxsDsm.total_files += 1

    @staticmethod
    def get_simplified_bmys():
        print('求出简化版品牌车型年款对照表...')
        bmy_set = set()
        sim_org_dict = {}
        org_sim_dict = {}
        raw_ds = './support/raw_bid_train_ds.txt'
        #raw_ds = './support/new_bid_train_ds.txt'
        with open(raw_ds, 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[1])
                bmy_set.add(bmy_id)
        print('共有{0}个年款！'.format(len(bmy_set)))
        lst = list(bmy_set)
        lst.sort()
        for idx, bmy_id in enumerate(lst):
            sim_org_dict[idx] = bmy_id
            org_sim_dict[bmy_id] = idx
            print('idx={0}:{1};'.format(idx, bmy_id))
        with open('./support/bmy_sim_org_dict.txt', 'w+', encoding='utf-8') as sofd:
            for kso, vso in sim_org_dict.items():
                sofd.write('{0}:{1}\n'.format(kso, vso))
        with open('./support/bmy_org_sim_dict.txt', 'w+', encoding='utf-8') as osfd:
            for kos, vos in org_sim_dict.items():
                osfd.write('{0}:{1}\n'.format(kos, vos))
        # 生成新的训练数据集
        WxsDsm.simplify_bid_ds(org_sim_dict, './support/bid_train_ds.txt', raw_ds)
        # 生成新的测试数据集
        #WxsDsm.simplify_bid_ds(org_sim_dict, './support/bid_test_ds.txt', './support/raw_bid_test_ds.txt')
        # 生成新寒武纪需要的标签文件
        WxsDsm.generate_cambricon_labels(sim_org_dict)
    
    @staticmethod
    def simplify_bid_ds(org_sim_dict, new_ds_file, org_ds_file):
        '''
        统计出数据集文件中不重复的bmy_id，然后重新从0开始递增编号作为新的
        bmy_id，并且维护两个bmy_id之间的对应关系，将旧的数据集文件中旧的
        bmy_id换为新的bmy_id
        '''
        with open(new_ds_file, 'w+', encoding='utf-8') as new_ds_fd:
            with open(org_ds_file, 'r', encoding='utf-8') as org_ds_fd:
                for line in org_ds_fd:
                    line = line.strip()
                    arrs0 = line.split('*')
                    org_bmy_id = int(arrs0[1])
                    img_file = arrs0[0]
                    bmy_id = org_sim_dict[org_bmy_id]
                    new_ds_fd.write('{0}*{1}\n'.format(img_file, bmy_id))

    @staticmethod
    def generate_cambricon_labels(sim_org_dict):
        '''
        由新的bmy_id求出老的bmy_id，然后求出品牌车型年款并用逗号分隔，生成一个txt文件
        '''
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        with open('./support/cambricon_vehicle_label.txt', 'w+', encoding='utf-8') as fd:
            for sim_bmy_id in range(len(sim_org_dict)):
                bmy_id = sim_org_dict[sim_bmy_id] + 1
                bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                bmy_name = bmy_vo['bmy_name']
                arrs0 = bmy_name.split('-')
                brand_name = arrs0[0]
                model_name = arrs0[1]
                year_name = arrs0[2]
                fd.write('{0},{1},{2},{3},{4},{5}\n'.format(
                    brand_name, model_name, year_name,
                    bmy_vo['brand_code'],
                    bmy_vo['model_code'],
                    bmy_vo['bmy_code']
                ))

    @staticmethod
    def get_fgvc_id_brand_dict():
        '''
        从logs/cambricon_vehicle_label.txt文件中，读出分类编号与
        品牌名称的对应关系，在求精度时，网络输出的fgvc_id求出品牌
        名称1，由正确答案fgvc_id求出品牌名称2，如果二者相等，则认
        为品牌预测正确，目前要求品牌识别正确率大于90%
        '''
        fgvc_id_brand_dict = {}
        fgvc_id = 0
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs0 = line.split(',')
                brand_name = arrs0[0]
                fgvc_id_brand_dict[fgvc_id] = brand_name
                fgvc_id += 1
        return fgvc_id_brand_dict

    @staticmethod
    def convert_to_brand_ds(bmy_ds_file, brand_ds_file, is_create_brands_dict=False):
        '''
        将年款数据集转换为品牌数据集
        参数：
            bmy_ds_file：年款数据集文件名
            brand_ds_file：品牌数据集文件名
        '''
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_id_brand_name_dict = {}
        brand_name_brand_id_dict = {}
        brand_set = set()
        if not is_create_brands_dict:
            print('read brand file')
            with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs0 = line.split(':')
                    brand_id = int(arrs0[0])
                    brand_name = arrs0[1]
                    brand_id_brand_name_dict[brand_id] = brand_name
                    brand_name_brand_id_dict[brand_name] = brand_id
                    brand_set.add(brand_name)
        idx = 0
        # 从精简bmy_id到原始bmy_id
        bmy_sim_org_dict = {}
        with open('./support/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
            for line in sofd:
                line = line.strip()
                arrs0 = line.split(':')
                bmy_sim_org_dict[int(arrs0[0])] = int(arrs0[1])
        with open(brand_ds_file, 'w+', encoding='utf-8') as bfd:
            with open(bmy_ds_file, 'r', encoding='utf-8') as yfd:
                for line in yfd:
                    line = line.strip()
                    arrs0 = line.split('*')
                    img_file = arrs0[0]

                    sim_bmy_id = int(arrs0[1])
                    #bmy_id = int(arrs0[1]) + 1
                    bmy_id = bmy_sim_org_dict[sim_bmy_id] + 1

                    bmy_name = bmy_id_bmy_name_dict[bmy_id]
                    arrs1= bmy_name.split('-')
                    brand_name = arrs1[0]
                    if not (brand_name in brand_set):
                        brand_set.add(brand_name)
                        brand_id_brand_name_dict[idx] = brand_name
                        brand_name_brand_id_dict[brand_name] = idx
                        idx += 1
                    brand_id = brand_name_brand_id_dict[brand_name]
                    bfd.write('{0}*{1}*{2}\n'.format(img_file, sim_bmy_id, brand_id))
        if is_create_brands_dict:
            with open('./support/bid_brands_dict.txt', 'w+', encoding='utf-8') as fd:
                for k, v in brand_id_brand_name_dict.items():
                    fd.write('{0}:{1}\n'.format(k, v))
        return len(brand_set)

    @staticmethod
    def convert_to_brand_ds_main():
        brand_num_train = WxsDsm.convert_to_brand_ds('./support/bid_train_ds.txt', 
                    './support/bid_brand_train_ds.txt', 
                    is_create_brands_dict=True)
        #brand_num_test = WxsDsm.convert_to_brand_ds('./support/bid_test_ds.txt', 
         #           './support/bid_brand_test_ds.txt', 
         #           is_create_brands_dict=False)
        brand_num_test = 0
        print('品牌种类：train={0} & test={1};'.format(brand_num_train, brand_num_test))

    @staticmethod
    def get_brand_bmy_num_from_ds():
        '''
        从训练数据集和测试数据集中求出品牌和年款数量
        '''
        bmy_set = set()
        brand_set = set()
        with open('./datasets/CUB_200_2011/anno/bid_brand_train_ds.txt', 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[1])
                bmy_set.add(bmy_id)
                brand_id = int(arrs0[2])
                brand_set.add(brand_id)
        print('品牌数量：{0};'.format(len(brand_set)))
        for brand in brand_set:
            print('### {0};'.format(brand))
        print('年款数量：{0};'.format(len(bmy_set)))
        for bmy in bmy_set:
            print('@@@ {0};'.format(bmy))

    @staticmethod
    def get_bmy_id_img_num():
        '''
        由samples.txt文件中，求出每个年款的图片数，并按图片数由
        少到多排序，并统计出不足100张图片的年款数
        '''
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        bmy_id_img_num_dict = {}
        with open('./logs/samples.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[-1])+1
                if bmy_id not in bmy_id_img_num_dict:
                    bmy_id_img_num_dict[bmy_id] = 1
                else:
                    bmy_id_img_num_dict[bmy_id] += 1
        lst = []
        for k, v in bmy_id_img_num_dict.items():
            lst.append((k, v))
        lst.sort(key=lambda x:x[1], reverse=False)
        let_5 = 0
        bmy_5 = []
        let_10 = 0
        bmy_10 = []
        let_100 = 0
        bmy_100 = []
        let_1000 = 0
        bt_1000 = 0
        for item in lst:
            print('@ {0}[{1}]: {2};'.format(bmy_id_bmy_name_dict[item[0]], item[0], item[1]))
            if item[1]<=5:
                let_5 += 1
                bmy_5.append(item[0])
            elif 5<item[1]<=10:
                let_10 += 1
                bmy_10.append(item[0])
            elif 10<item[1]<=100:
                let_100 += 1
                bmy_100.append(item[0])
            elif 100<item[1]<=1000:
                let_1000 += 1
            else:
                bt_1000 += 1
        bmy_img_let_5_file = './logs/bmy_img_let_5.txt'
        print('共有{0}个年款小于5张图片；见文件：{1};'.format(let_5, bmy_img_let_5_file))
        WxsDsm.write_list_to_file(bmy_id_bmy_name_dict, bmy_img_let_5_file, bmy_5)
        bmy_img_let_10_file = './logs/bmy_img_let_10.txt'
        print('共有{0}个年款小于10张图片；见文件：{1}'.format(let_10, bmy_img_let_10_file))
        WxsDsm.write_list_to_file(bmy_id_bmy_name_dict, bmy_img_let_10_file, bmy_10)
        bmy_img_let_100_file = './logs/bmy_img_let_100.txt'
        print('共有{0}个年款小于100张图片；见文件：{1}'.format(let_100, bmy_img_let_100_file))
        WxsDsm.write_list_to_file(bmy_id_bmy_name_dict, bmy_img_let_100_file, bmy_100)
        print('共有{0}个年款小于1000张图片;'.format(let_1000))
        print('共有{0}个年款大于1000张图片;'.format(bt_1000))


    @staticmethod
    def write_list_to_file(bmy_id_bmy_name_dict, filename, lst):
        with open(filename, 'w+', encoding='utf-8') as wfd:
            for item in lst:
                bmy_name = bmy_id_bmy_name_dict[int(item)]
                wfd.write('{0}: {1}\n'.format(item, bmy_name))

    @staticmethod
    def get_current_state():
        bid_brand_set, bid_model_set, bid_bmy_set, bid_vin_set, _, _, _, _ = WxsDsm._get_bid_info()
        print('标书要求：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(bid_vin_set), len(bid_brand_set), len(bid_bmy_set)
        ))
        # 统计当前情况
        curr_brand_set, curr_bmy_set = WxsDsm.get_current_info()
        delta_brand = bid_brand_set - curr_brand_set
        delta_bmy = bid_bmy_set - curr_bmy_set
        print('缺失品牌数量为{0}个，分别为：'.format(len(delta_brand)))
        for brand in delta_brand:
            print('# {0};'.format(brand))
        print('缺失年款数量为{0};'.format(len(delta_bmy)))
        #
        unknown_vin_codes, new_brand_set, new_bmy_set, brand_vins_dict, bmy_vins_dict \
                    = WxsDsm.get_g2n_vin_codes(
                        bid_brand_set, bid_bmy_set, 
                        curr_brand_set, curr_bmy_set
                    )
        delta_brand0 = new_brand_set - curr_brand_set
        delta_brand =  delta_brand0 & bid_brand_set
        print('需处理品牌数量为{0}个;'.format(len(delta_brand)))
        delta_bmy0 = new_bmy_set - curr_bmy_set
        delta_bmy = delta_bmy0 & bid_bmy_set
        print('需处理年款数量为{0}个;'.format(len(delta_bmy)))
        to_be_processed_vins = set()
        for dbi in delta_brand:
            for bvc in brand_vins_dict[dbi]:
                to_be_processed_vins.add(bvc)
        for dmi in delta_bmy:
            for mvc in bmy_vins_dict[dmi]:
                to_be_processed_vins.add(mvc)
        print('需要添加的车辆识别码数量为{0};'.format(len(to_be_processed_vins)))
        '''
        tbp_lst = list(to_be_processed_vins)
        tbp_lst.sort()
        for idx in range(9):
            print('idx={0};'.format(idx))
            dst_base_folder = '/media/zjkj/work/abc/g2n/task{0}'.format(idx+1)
            os.mkdir(dst_base_folder)
            if (idx+1)*173 > len(tbp_lst):
                task = tbp_lst[idx*173:]
            else:
                task = tbp_lst[idx*173:(idx+1)*173]
            print('task.len={0};'.format(len(task)))
            for ti in task:
                src_path = Path('/media/zjkj/work/guochanche_2n/{0}'.format(ti))
                dst_folder = '/media/zjkj/work/abc/g2n/task{0}/{1}'.format(idx+1, ti)
                os.mkdir(dst_folder)
                print('dst_folder:{0};'.format(dst_folder))
                for fi in src_path.iterdir():
                    fi_str = str(fi)
                    arrs0 = fi_str.split('/')
                    filename = arrs0[-1]
                    shutil.copy(fi_str, '{0}/{1}'.format(dst_folder, filename))
        '''
        with open('./logs/vins_to_be_processed.txt', 'w+', encoding='utf-8') as vfd:
            for vcv in to_be_processed_vins:
                vfd.write('{0}\n'.format(vcv))
        with open('./logs/new_added_brand.txt', 'w+', encoding='utf-8') as bfd:
            for bn in delta_brand:
                bfd.write('{0}\n'.format(bn))
        with open('./logs/unknown_vins_20200718.txt', 'w+', encoding='utf-8') as ufd:
            for uv in unknown_vin_codes:
                ufd.write('{0}\n'.format(uv))
        now_brand = curr_brand_set | delta_brand
        now_lack_brand = bid_brand_set - now_brand
        now_lack_brand_txt = './logs/now_lack_brand.txt'
        print('现在缺少品牌数量为：{0}个；见{1};'.format(len(now_lack_brand), now_lack_brand_txt))
        with open(now_lack_brand_txt, 'w+', encoding='utf-8') as fd1:
            for nlb in now_lack_brand:
                fd1.write('{0}\n'.format(nlb))
        now_bmy = curr_bmy_set | delta_bmy
        now_lack_bmy = bid_bmy_set - now_bmy
        now_lack_bmy_txt = './logs/now_lack_bmy.txt'
        print('现在缺少年款数量为：{0}个，见{1};'.format(len(now_lack_bmy), now_lack_bmy_txt))
        with open(now_lack_bmy_txt, 'w+', encoding='utf-8') as fd2:
            for nlm in now_lack_bmy:
                fd2.write('{0}\n'.format(nlm))

    @staticmethod
    def get_current_info():
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        bmy_set = set()
        with open('./logs/samples.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[1]) + 1
                bmy_name = bmy_id_bmy_name_dict[bmy_id]
                bmy_set.add(bmy_name)
                arrs1 = bmy_name.split('-')
                brand_name = arrs1[0]
                brand_set.add(brand_name)
        return brand_set, bmy_set

    @staticmethod
    def get_g2n_vin_codes(bid_brand_set, bid_bmy_set, curr_brand_set, curr_bmy_set):
        vin_code_bmy_id_dict = CBmy.get_vin_code_bmy_id_dict()
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        base_path = Path('//media/zjkj/work/guochanche_2n')
        unknown_vin_codes = []
        vin_codes = []
        num = 0
        num_new_vc = 0
        num_new_bmy = 0
        new_bmy_set = set()
        new_brand_set = set()
        brand_vins_dict = {}
        bmy_vins_dict = {}
        for vc in base_path.iterdir():
            vc_str = str(vc)
            arrs0 = vc_str.split('/')
            vin_code = arrs0[-1]
            if vin_code not in vin_code_bmy_id_dict:
                #vin_codes.append(vin_code)
                unknown_vin_codes.append(vin_code)
                num_new_vc += 1
                print('   add case 1: {0};'.format(vin_code))
            else:
                bmy_id = vin_code_bmy_id_dict[vin_code]
                bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                bmy_name = bmy_vo['bmy_name']
                if bmy_name not in curr_bmy_set:
                    vin_codes.append(vin_code)
                    num_new_bmy += 1
                    arrs2 = bmy_name.split('-')
                    brand_name = arrs2[0]
                    new_brand_set.add(brand_name)
                    new_bmy_set.add(bmy_name)
                    print('   add case 2: {0};'.format(vin_code))
                    #
                    if brand_name not in brand_vins_dict:
                        brand_vins_dict[brand_name] = [vin_code]
                    else:
                        brand_vins_dict[brand_name].append(vin_code)
                    #
                    if bmy_name not in bmy_vins_dict:
                        bmy_vins_dict[bmy_name] = []
                    bmy_vins_dict[bmy_name].append(vin_code)
            num += 1
            if num % 100 == 0:
                print('已处理：{0}个...'.format(num))
        print('v1 新车辆识别码{0}个，新年款{1}个;'.format(num_new_vc, num_new_bmy))
        print('v1 增加的品牌数{0}个；新增加的年款数：{1}个;'.format(len(new_brand_set), len(new_bmy_set)))
        return unknown_vin_codes, new_brand_set, new_bmy_set, brand_vins_dict, bmy_vins_dict
        
    @staticmethod
    def generate_zjkj_fds_labels():
        '''
        将品牌车型年款信息由Cambricon格式标签文件改为公司要求格式：
        {"品牌编号", "车型编号", "年款编号", "品牌-车型-年款"},{......},
        ......
        每行有两个元素
        '''
        bmys = CBmy.get_zjkj_bmys()
        row_num = len(bmys) + 1
        print('从数据库中读出内容...')
        item_sep = ','
        with open('./support/zjkj_label_fds.txt', 'w+', encoding='utf-8') as zfd:
            zfd.write('{{"{0}", "{1}", "{2}", "{3}-{4}-{5}"}}{6}{7}'.format(
                    'b0000', 'b000011111', 'b00001111122222',
                    '未知品牌', '未知车型', '未知年款',
                    ',', ''
                ))
            row = 1
            for bmy in bmys:
                line_break = ''
                item_sep = ','
                if row % 2 != 0:
                    line_break = '\n'
                row += 1
                if row == row_num:
                    item_sep = ''
                zfd.write('{{"{0}", "{1}", "{2}", "{3}-{4}-{5}"}}{6}{7}'.format(
                    bmy['brand_code'], bmy['model_code'], bmy['bmy_code'],
                    bmy['brand_name'], bmy['model_name'], bmy['bmy_name'],
                    item_sep, line_break
                ))
    
        
    @staticmethod
    def generate_zjkj_cambricon_labels():
        '''
        将品牌车型年款信息由Cambricon格式标签文件改为公司要求格式：
        {"品牌编号", "车型编号", "年款编号", "品牌-车型-年款"},{......},
        ......
        每行有两个元素
        '''
        row = 0
        row_num = 0
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                row_num += 1
        item_sep = ','
        with open('./support/zjkj_label_v1.txt', 'w+', encoding='utf-8') as zfd:
            with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
                for line in cfd:
                    line = line.strip()
                    arrs0 = line.split(',')
                    brand_name = arrs0[0]
                    model_name = arrs0[1]
                    year_name = arrs0[2]
                    brand_code = arrs0[3]
                    model_code = arrs0[4]
                    year_code = arrs0[5]
                    line_break = ''
                    if row % 2 != 0:
                        line_break = '\n'
                    row += 1
                    if row == row_num:
                        item_sep = ''
                    zfd.write('{{"{0}", "{1}", "{2}", "{3}-{4}-{5}"}}{6}{7}'.format(
                        brand_code, model_code, year_code,
                        brand_name, model_name, year_name,
                        item_sep, line_break
                    ))
                
    @staticmethod
    def get_fine_wxs_dataset():
        '''
        获取所里筛查正确的数据集
        '''
        num = 0
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
        dst_folder = '/media/zjkj/work/fgvc_dataset/wxs/tds'
        test_files = []
        for brand_path in base_path.iterdir():
            for model_path in brand_path.iterdir():
                for year_path in model_path.iterdir():
                    for file_path in year_path.iterdir():
                        file_str = str(file_path)
                        arrs0 = file_str.split('/')
                        file_name = arrs0[-1]
                        num += 1
                        if num % 1000 == 0:
                            print('处理完成{0}个文件！'.format(num))
                        if not file_path.is_dir() and file_name.endswith(('jpg', 'png', 'bmp', 'jpeg')) \
                                    and (file_name.startswith('白') or file_name.startswith('夜')):
                            test_files.append(file_str)
        bmy_name_bmy_id_dict = CBmy.get_bmy_name_bmy_id_dict()
        with open('./logs/wxs_tds.csv', 'w+', encoding='utf-8') as tfd:
            for tf in test_files:
                arrs0 = tf.split('/')
                file_name = arrs0[-1]
                year_name = arrs0[-2].replace('-', '_')
                model_name = arrs0[-3].replace('-', '_')
                brand_name = '{0}牌'.format(arrs0[-4])
                brand_folder = '{0}/{1}'.format(dst_folder, brand_name)
                if not os.path.exists(brand_folder):
                    os.mkdir(brand_folder)
                model_folder = '{0}/{1}'.format(brand_folder, model_name)
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                year_folder = '{0}/{1}'.format(model_folder, year_name)
                if not os.path.exists(year_folder):
                    os.mkdir(year_folder)
                shutil.copy(tf, '{0}/{1}'.format(year_folder, file_name))
                bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                if bmy_name in bmy_name_bmy_id_dict:
                    bmy_id = bmy_name_bmy_id_dict[bmy_name]
                else:
                    bmy_id = 0
                print('{0}*{1}'.format(tf, bmy_id-1))
                tfd.write('{0}/{1}/{2}/{3}/{4},{5}\n'.format(dst_folder, brand_name, model_name, year_name, file_name, bmy_id-1))
        print('共有{0}个测试集文件！'.format(len(test_files)))

    @staticmethod
    def generate_wxs_bmy_csv():
        recs = CBmy.get_bmys()
        with open('./logs/wxs_bmy_csv.csv', 'w+', encoding='utf-8') as cfd:
            for rec in recs:
                bmy_type = '所里'
                if rec['bmy_code'].startswith('b'):
                    bmy_type = '我们'
                cfd.write('{0},{1}, {2}\n'.format(rec['bmy_id'], rec['bmy_name'], bmy_type))

    @staticmethod
    def generate_error_samples_html():
        '''
        将验证模型时找出预测错误的样本文件，转为HTML文件，可以依次
        浏览图片文件，查看正确结果和网络输出，便于分析出错原因
        images/
        index.html
        '''
        num = 1
        '''
        # 品牌车型年款为主时使用
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        bmy_sim_org_dict = {}
        with open('./support/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
            for line in sofd:
                line = line.strip()
                arrs0 = line.split(':')
                sim_id = int(arrs0[0])
                org_id = int(arrs0[1])
                bmy_sim_org_dict[sim_id] = org_id
        '''
        # 品牌为主时使用
        bid_brand_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                bid_brand_dict[int(arrs0[0])] = arrs0[1]
        with open('../work/es0906/index.html', 'w+', encoding='utf-8') as hfd:
            hfd.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>错误分类结果分析页面</title>
<script type="text/javascript">
let images = [
""")
            with open('./logs/top1_error_samples.txt', 'r', encoding='utf-8') as sfd:
                for line in sfd:
                    line = line.strip()
                    arrs0 = line.split('*')
                    full_file = arrs0[0]
                    arrs1 = full_file.split('/')
                    img_file = arrs1[-1]
                    dst_img_file = 'images/{0:05d}.jpg'.format(num)
                    dst_full_file = '../work/es0906/images/{0:05d}.jpg'.format(num)
                    print('拷贝文件：{0};'.format(dst_full_file))
                    shutil.copy(full_file, dst_full_file)
                    num += 1
                    '''
                    # 品牌车型年款为主时使用
                    gt_id = bmy_sim_org_dict[int(arrs0[1])]
                    gt_label = bmy_id_bmy_name_dict[gt_id+1]
                    net_id = bmy_sim_org_dict[int(arrs0[2])]
                    net_label = bmy_id_bmy_name_dict[net_id+1]
                    '''
                    # 品牌为主时使用
                    gt_id = int(arrs0[1])
                    gt_label = bid_brand_dict[gt_id]
                    net_id = int(arrs0[2])
                    net_label = bid_brand_dict[net_id]
                    item = {
                        'orgFile': full_file,
                        'imgFile': dst_img_file,
                        'gtLabel': gt_label,
                        'netLabel': net_label
                    }
                    hfd.write('{0},'.format(item))
            # 写下后面的代码
            hfd.write("""
]
let g_idx = 0

function showPage() {
    let orgFile = document.getElementById("orgFile")
    orgFile.innerText = images[g_idx].orgFile
	let vehicleImg = document.getElementById("vehicleImg")
	vehicleImg.src = images[g_idx].imgFile
	let gtLabel = document.getElementById("gtLabel")
	gtLabel.innerText = images[g_idx].gtLabel
	let netLabel = document.getElementById("netLabel")
	netLabel.innerText = images[g_idx].netLabel
	let currPage = document.getElementById("currPage")
	currPage.value = "" + (g_idx + 1)
}

function prevImg() {
	g_idx--
	if (g_idx <= 0) {
		g_idx = 0
	}
	showPage()
}

function goPage() {
	let currPage = document.getElementById("currPage")
	let pageNum = parseInt(currPage.value) - 1
	if (pageNum <= 0) {
		g_idx = 0
	} else if (pageNum >= images.length-1) {
		g_idx = images.length-1
	} else {
		g_idx = pageNum
	}
	showPage()
}

function nextImg() {
	g_idx++
	if (g_idx >= images.length-1) {
		g_idx = images.length-1
	}
	showPage()
}
</script>
</head>
 
<body onLoad="showPage()">
原始文件：<span id="orgFile"></span><br />
<img id="vehicleImg" src="images/000001.jpg" style="height: 500px;" /><br />
正确结果：<span id="gtLabel"></span><br />
预测结果：<span id="netLabel"></span><br />
<input type="button" value="上一张" onClick="prevImg()" />
<input type="text" id="currPage" value="1" />
<input type="button" value="跳转" onClick="goPage()" />
<input type="button" value="下一张" onClick="nextImg()" />
</body>
 
</html>
""")

    @staticmethod
    def generate_test_ds_bmy_csv():
        bmy_name_bmy_id_dict = CBmy.get_bmy_name_bmy_id_dict()
        dst_folder = '/media/zjkj/work/fgvc_dataset/wxs/fine'
        with open('./logs/wxs_tsd_v2.csv', 'w+', encoding='utf-8') as tfd:
            with open('./datasets/CUB_200_2011/anno/test_ds_v4.txt', 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs = line.split('*')
                    tf = arrs[0]
                    arrs0 = tf.split('/')
                    file_name = arrs0[-1]
                    year_name = arrs0[-2].replace('-', '_')
                    model_name = arrs0[-3].replace('-', '_')
                    brand_name = '{0}牌'.format(arrs0[-4])
                    brand_folder = '{0}/{1}'.format(dst_folder, brand_name)
                    if not os.path.exists(brand_folder):
                        os.mkdir(brand_folder)
                    model_folder = '{0}/{1}'.format(brand_folder, model_name)
                    if not os.path.exists(model_folder):
                        os.mkdir(model_folder)
                    year_folder = '{0}/{1}'.format(model_folder, year_name)
                    if not os.path.exists(year_folder):
                        os.mkdir(year_folder)
                    shutil.copy(tf, '{0}/{1}'.format(year_folder, file_name))
                    bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                    if bmy_name in bmy_name_bmy_id_dict:
                        bmy_id = bmy_name_bmy_id_dict[bmy_name]
                    else:
                        bmy_id = 0
                    print('{0}*{1}'.format(tf, bmy_id-1))
                    tfd.write('{0}/{1}/{2}/{3}/{4},{5}\n'.format(dst_folder, brand_name, model_name, year_name, file_name, bmy_id-1))
        
    @staticmethod
    def copy_test_ds_images_for_cnstream():
        dst_folder = '/media/zjkj/work/repository/cnstream/images'
        idx = 0
        with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds.txt', 'r', encoding='utf-8') as tsfd:
            for line in tsfd:
                line = line.strip()
                arrs0 = line.split('*')
                src_file = arrs0[0]
                arrs1 = src_file.split('/')
                img_file = arrs1[-1]
                idx += 1
                full_str = '{0:06d}'.format(idx)
                folder0 = '{0}/{1}'.format(dst_folder, full_str[:2])
                if not os.path.exists(folder0):
                    os.mkdir(folder0)
                folder1 = '{0}/{1}'.format(folder0, full_str[2:4])
                if not os.path.exists(folder1):
                    os.mkdir(folder1)
                shutil.copy(src_file, '{0}/{1}'.format(folder1, img_file))
                print('copy{0}: {1} => {2};'.format(idx, src_file, '{0}/{1}'.format(folder1, img_file)))
    
    @staticmethod
    def integrate_wxs_test_ds():
        num = 0
        with open('./logs/wxs_test_ds.txt', 'w+', encoding='utf-8') as dfd:
            with open('./logs/wxs_tds_v1.csv', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs0 = line.split(',')
                    file_str = arrs0[0]
                    bmy_id = int(arrs0[1]) + 1
                    if bmy_id <= 0:
                        num += 1
                    else:
                        print('{0}*{1}'.format(file_str, (bmy_id - 1)))
                        dfd.write('{0}*{1}\n'.format(file_str, (bmy_id - 1)))
        print('缺失年款数：{0};'.format(num))

    @staticmethod
    def generate_vin_bmy_csv():
        recs = CBmy.get_vin_code_bmys()
        rows = []
        source_type = ''
        for rec in recs:
            bmy_id = int(rec['bmy_id'])
            bmy_vo = CBmy.get_bmy_by_id(bmy_id)
            if int(rec['source_type']) == 1:
                source_type = '所里'
            else:
                source_type = '我们'
            row = [source_type, bmy_vo['bmy_name'], rec['vin_code']]
            rows.append(row)
        with open('./logs/wxs_vin_code_bmy_check.csv', 'w+', encoding='utf-8') as fd:
            for row in rows:
                fd.write('{0},{1},{2}\n'.format(row[0], row[1], row[2]))

    @staticmethod
    def exp001_1():
        bmy_set = set()
        num = 0
        with open('./support/bid_train_ds.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[-1])
                bmy_set.add(bmy_id)
                num += 1
                if num % 1000 == 0:
                    print('处理{0}条记录;'.format(num))
        print('共有{0}个年款'.format(len(bmy_set)))
        bmy_id_brand_id_dict = CBmy.get_bmy_id_brand_id_dict()
        bmy_sim_org_dict = {}
        with open('./support/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
            for line in sofd:
                line = line.strip()
                arrs0 = line.split(':')
                bmy_sim_org_dict[int(arrs0[0])] = int(arrs0[1])
        brand_set = set()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        for bi in bmy_set:
            bmy_id = bmy_sim_org_dict[bi] + 1
            bmy_name = bmy_id_bmy_name_dict[bmy_id]
            arrs0 = bmy_name.split('-')
            brand_name = arrs0[0]
            #brand_id = bmy_id_brand_id_dict[bmy_id]
            brand_set.add(brand_name)
        print('共有{0}品牌'.format(len(brand_set)))
        blst = list(brand_set)
        blst.sort()
        with open('./support/b1.txt', 'w+', encoding='utf-8') as b1fd:
            for idx, bl in enumerate(blst):
                print('{0}: {1};'.format(idx, bl))
                b1fd.write('{0}: {1}\n'.format(idx, bl))
        b2 = []
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as fd1:
            for line in fd1:
                line = line.strip()
                arrs0 = line.split(':')
                b2.append(arrs0[1])
        b2.sort()
        with open('./support/b2.txt', 'w+', encoding='utf-8') as b2fd:
            for idx, b2i in enumerate(b2):
                print('### {0}: {1};'.format(idx, b2i))
                b2fd.write('{0}: {1}\n'.format(idx, b2i))

    @staticmethod
    def process_unknown_wxs_tds():
        '''
        处理无锡所测试集中，没有错误的5664张图片中，新车型和新年款的记录，
        将其增加新车型和新年款，并形成到测试集中：
        1. 添加到zjkj_label_v1.txt中；
        2. 添加到bid_brand_test_ds.txt和bid_brand_train_ds.txt文件中
        '''
        brand_set, brand_name_code_dict, bm_set, bm_name_code_dict, bmy_set, bmy_name_code_dict = WxsDsm.get_bdb_from_cambricon_label()
        print('现有品牌{0}个'.format(len(brand_set)))
        print('现有车型{0}个'.format(len(bm_set)))
        print('现有年款{0}个'.format(len(bmy_set)))
        w_brand_set, w_brand_name_code_dict, w_bm_set, w_bm_name_code_dict, w_bmy_set, w_bmy_name_code_dict = WxsDsm.wxs_excel_bmy_data()
        
        t_brand_set = set()
        t_bm_set = set()
        t_bmy_set = set()
        with open('./logs/wxs_tds_0730.csv', 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs0 = line.split(',')
                arrs1 = arrs0[0].split('/')
                img_file = arrs1[-1]
                arrs2 = img_file.split('_')
                brand_name = arrs2[3].replace('-', '_')
                t_brand_set.add('{0}牌'.format(brand_name))
                model_name = arrs2[4].replace('-', '_')
                bm_name = '{0}牌-{1}'.format(brand_name, model_name)
                t_bm_set.add(bm_name)
                year_name = arrs2[5].replace('-', '_')
                bmy_name = '{0}牌-{1}-{2}'.format(brand_name, model_name, year_name)
                t_bmy_set.add(bmy_name)
                sim_bmy_id = int(arrs0[1])
                if sim_bmy_id < 0:
                    print('{0}: {1};'.format(img_file, bmy_name))
        
        diff_brand = t_brand_set - w_brand_set
        print('测试集共有{0}个品牌，增加的新品牌{1}个，标书品牌{2}个'.format(len(t_brand_set), len(diff_brand), len(w_brand_set)))
        for bi in diff_brand:
            print(bi)
        

    @staticmethod
    def get_bdb_from_cambricon_label():
        brand_set = set()
        brand_name_code_dict = {}
        bm_set = set()
        bm_name_code_dict = {}
        bmy_set = set()
        bmy_name_code_dict = {}
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                arrs0 = line.split(',')
                # 处理品牌
                brand_name = arrs0[0]
                brand_code = arrs0[3]
                brand_set.add(brand_name)
                if brand_name not in brand_name_code_dict:
                    brand_name_code_dict[brand_name] = brand_code
                # 处理车型
                model_name = arrs0[1]
                bm_code = arrs0[4]
                bm_name = '{0}-{1}'.format(brand_name, model_name)
                bm_set.add(bm_name)
                if bm_name not in bm_name_code_dict:
                    bm_name_code_dict[bm_name] = bm_code
                # 处理年款
                year_name = arrs0[2]
                bmy_code = arrs0[5]
                bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                bmy_set.add(bmy_name)
                if bmy_name not in bmy_name_code_dict:
                    bmy_name_code_dict[bmy_name] = bmy_code
        return brand_set, brand_name_code_dict, bm_set, bm_name_code_dict, bmy_set, bmy_name_code_dict

    @staticmethod
    def wxs_excel_bmy_data():
        brand_set = set()
        brand_name_code_dict = {}
        bm_set = set()
        bm_name_code_dict = {}
        bmy_set = set()
        bmy_name_code_dict = {}
        first_row = True
        with open('./logs/bid_20200708.csv', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                if first_row:
                    first_row = False
                    continue
                arrs0 = line.split(',')
                # 处理品牌
                brand_code = arrs0[1]
                brand_name = arrs0[2]
                brand_set.add(brand_name)
                if brand_name not in brand_name_code_dict:
                    brand_name_code_dict[brand_name] = brand_code
                # 处理车型
                bm_code = arrs0[3]
                model_name = arrs0[4]
                bm_name = '{0}-{1}'.format(brand_name, model_name)
                bm_set.add(bm_name)
                if bm_name not in bm_name_code_dict:
                    bm_name_code_dict[bm_name] = bm_code
                # 处理年款
                bmy_code = arrs0[5]
                year_name = arrs0[6]
                bmy_name = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                bmy_set.add(bmy_name)
                if bmy_name not in bmy_name_code_dict:
                    bmy_name_code_dict[bmy_name] = bmy_code
        return brand_set, brand_name_code_dict, bm_set, bm_name_code_dict, bmy_set, bmy_name_code_dict

    @staticmethod
    def fix_test_ds_brand_errors():
        '''
        将子品牌合并为主品牌
        '''
        bid_brands_dict = {}
        brand_id_brand_name_dict = {}
        brand_name_brand_id_dict = {}
        error_num = 0
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                bid_brands_dict[int(arrs0[0])] = arrs0[1]
                brand_id_brand_name_dict[int(arrs0[0])] = arrs0[1]
                brand_name_brand_id_dict[arrs0[1]] = int(arrs0[0])
        with open('./logs/n_bid_brand_test_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds.txt', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs0 = line.split('*')
                    full_fn = arrs0[0]
                    arrs1 = full_fn.split('/')
                    img_file = arrs1[-1]
                    arrs2 = img_file.split('_')
                    file_brand_name = '{0}牌'.format(arrs2[3])
                    sim_bmy_id = arrs0[1]
                    sim_brand_id = int(arrs0[2])
                    if sim_brand_id == 210:
                        sim_brand_id = 29
                    if file_brand_name == 'JEEP牌':
                        file_brand_name = '吉普牌'
                    if file_brand_name == '广汽传祺牌':
                        file_brand_name = '广汽牌'
                        sim_brand_id = 100
                    if file_brand_name == '荣威牌':
                        sim_brand_id = 71
                    if file_brand_name == '北汽绅宝牌':
                        file_brand_name = '北京牌'
                        sim_brand_id = 26
                    if file_brand_name == '莲花牌':
                        sim_brand_id = 96
                    if file_brand_name == '吉利全球鹰牌':
                        file_brand_name = '吉利牌'
                        sim_brand_id = 114
                    if file_brand_name == '双环牌':
                        sim_brand_id = 101
                    if file_brand_name == '雪铁龙牌':
                        sim_brand_id = 58
                    if file_brand_name == '哈弗牌':
                        sim_brand_id = 60
                    if file_brand_name == '悍马牌':
                        sim_brand_id = 5
                    if file_brand_name == '别克牌':
                        sim_brand_id = 91
                    if file_brand_name == '大众牌':
                        sim_brand_id = 37
                    if file_brand_name == '金杯牌':
                        sim_brand_id = 102
                    if file_brand_name == '斯巴鲁牌':
                        sim_brand_id = 123
                    if file_brand_name == '福特牌':
                        sim_brand_id = 2
                    if file_brand_name == '奔驰牌':
                        sim_brand_id = 6
                    if file_brand_name == '一汽奔腾牌':
                        file_brand_name = '一汽牌'
                        sim_brand_id = 23
                    if file_brand_name == '双龙牌':
                        file_brand_name = '双龙大宇牌'
                        sim_brand_id = 145
                    if file_brand_name == '长安商用牌':
                        file_brand_name = '长安牌'
                        sim_brand_id = 16
                    if file_brand_name == '东风风光牌':
                        file_brand_name = '东风牌'
                        sim_brand_id = 69
                    ds_brand_name = bid_brands_dict[sim_brand_id]
                    if file_brand_name != ds_brand_name:
                        error_num += 1
                        print('{0}: {1} vs {2};'.format(img_file, file_brand_name, ds_brand_name))
                    else:
                        print('{0}*{1}*{2}'.format(img_file, sim_bmy_id, sim_brand_id))
                        wfd.write('{0}*{1}*{2}\n'.format(full_fn, sim_bmy_id, sim_brand_id))

    @staticmethod
    def get_wxs_bmys():
        '''
        从数据库中读出品牌车型年款列表
        '''
        bmys = CBmy.get_wxs_bmys()
        with open('./logs/wxs_bmys.csv', 'w+', encoding='utf-8') as bfd:
            for bi in bmys:
                print(bi)
                bfd.write('{0},{1},{2}\n'.format(bi['bmy_id'], bi['bmy_code'], bi['bmy_name']))

    @staticmethod
    def get_non_wxs_vins():
        '''
        从t_vin表中获取当前不在所里5731个品牌车型年款中的车辆识别码
        '''
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        vins = CBmy.get_non_wxs_vins()
        vin_img_file_dict = {}
        num = 0
        missing_num = 0
        missing_vins = []
        # 从samples.txt中读出文件名
        with open('./support/samples.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs0 = line.split('*')
                full_fn = arrs0[0]
                arrs1 = full_fn.split('/')
                img_file = arrs1[-1]
                raw_vin_code = img_file.split('_')
                arrs2 = raw_vin_code[0].split('#')
                vc = arrs2[0]
                vin_img_file_dict[vc] = full_fn
                num += 1
                if num % 1000 == 0:
                    print('处理{0}条样本数据...'.format(num))
        # 处理guochanche_2n目录内容
        new_vin_folder_dict = {}
        base_path = Path('/media/zjkj/work/guochanche_2n')
        for path_obj in base_path.iterdir():
            path_str = str(path_obj)
            arrs0 = path_str.split('/')
            vin_code = arrs0[-1]
            new_vin_folder_dict[vin_code] = path_str
        # 处理每个车辆识别码
        with open('./support/to_be_processed_vins.csv', 'w+', encoding='utf-8') as vfd:
            for vin in vins:
                print(vin)
                bmy_id = int(vin['bmy_id'])
                bmy_name = bmy_id_bmy_vo_dict[bmy_id]['bmy_name']
                if vin['vin_code'] in vin_img_file_dict:
                    img_file = vin_img_file_dict[vin['vin_code']]
                elif vin['vin_code'] in new_vin_folder_dict:
                    new_path = Path(new_vin_folder_dict[vin['vin_code']])
                    for file_obj in new_path.iterdir():
                        img_file = str(file_obj)
                        break
                else:
                    img_file = '?????????'
                    missing_num += 1
                    missing_vins.append({
                        'vin_id': vin['vin_id'],
                        'bmy_name':bmy_name,
                        'vin_code': vin['vin_code']
                    })
                print('{0},{1},{2},{3}'.format(vin['vin_id'], bmy_name, vin['vin_code'], img_file))
                vfd.write('{0},{1},{2},{3}\n'.format(vin['vin_id'], bmy_name, vin['vin_code'], img_file))
        print('共{0}个车辆识别码未找到图片'.format(missing_num))
        for mv in missing_vins:
            print('### {0};'.format(mv))


    @staticmethod
    def generate_wxs_tds_table():
        '''
        将无锡所测试文件放到Excel表格中
        '''
        num = 0
        base_path = Path('/media/zjkj/work/品牌')
        with open('./support/wxs_tds_images.csv', 'w+', encoding='utf-8') as tfd:
            for path_obj in base_path.iterdir():
                for file_obj in path_obj.iterdir():
                    full_fn = str(file_obj)
                    arrs0 = full_fn.split('/')
                    img_file = arrs0[-1]
                    parent_folder = arrs0[-2]
                    if file_obj.is_file() and img_file.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                        print('./{0}/{1}'.format(parent_folder, img_file))
                        tfd.write('{0}/{1}, \n'.format(parent_folder, img_file))
                        num += 1
        print('共有{0}个图片文件'.format(num))

    @staticmethod
    def get_wxs_vin_id_img_num():
        '''
        求出无锡所Excel表格中每个车辆识别码的图片数，并列出图片数为零的
        车辆识别码编号
        '''
        vin_img_num_dict = WxsDsm.get_vin_img_num_dict()
        vins = CBmy.get_wxs_vins()
        wxs_vin_imgs_dict = {}
        empty_wxs_vins = []
        # 精确匹配
        accurate_match_num = 0
        accurate_vin_img_num_dict = {}
        # 模糊查询
        fuzzy_match_num = 0
        fuzzy_vin_img_num_dict = {}
        for vin in vins:
            if vin['vin_code'] in vin_img_num_dict:
                print('精确匹配 vin_code: {0};'.format(vin['vin_code']))
                wxs_vin_imgs_dict[vin['vin_code']] = vin_img_num_dict[vin['vin_code']]
                accurate_match_num += 1
                accurate_vin_img_num_dict[vin['vin_code']] = vin_img_num_dict[vin['vin_code']]
            else:
                contained = False
                for k, v in vin_img_num_dict.items():
                    if k.startswith(vin['vin_code']):
                        print('模糊匹配 vin_code: {0}; [{1}]'.format(vin['vin_code'], k))
                        wxs_vin_imgs_dict[vin['vin_code']] = vin_img_num_dict[k]
                        contained = True
                        fuzzy_match_num += 1
                        fuzzy_vin_img_num_dict[vin['vin_code']] = vin_img_num_dict[k]
                        break
                if not contained:
                    print('##### 未找到 vin_code: {0}; [{1}]'.format(vin['vin_code'], k))
                    wxs_vin_imgs_dict[vin['vin_code']] = 0
                    empty_wxs_vins.append(vin['vin_code'])
        print('共有{0}个车辆识别码，其中{1}个为空；精确匹配{2}个，模糊匹配{3}个'\
            .format(len(vins), len(empty_wxs_vins), accurate_match_num, fuzzy_match_num))
        sorted_vid = sorted(wxs_vin_imgs_dict.items(), key=lambda x: x[1])
        with open('./support/wxs_vin_imgs.txt', 'w+', encoding='utf-8') as wfd:
            for vid in sorted_vid:
                wfd.write('{0}:{1}\n'.format(vid[0], vid[1]))
        with open('./support/wxs_empty_vins.txt', 'w+', encoding='utf-8') as efd:
            for vi in empty_wxs_vins:
                efd.write('{0}\n'.format(vi))

    @staticmethod
    def get_vin_img_num_dict():
        dict_file = './support/vin_img_num_dict.txt'
        vin_img_num_dict = {}
        if not os.path.exists(dict_file):
            # 统计进口车车辆识别码和图片数量
            WxsDsm.get_import_vehicle_vin_set_img_num(vin_img_num_dict)
            WxsDsm.get_domestic_vehicle_vin_set_img_num(vin_img_num_dict)
            print('共有{0}条记录'.format(len(vin_img_num_dict.keys())))
            with open(dict_file, 'w+', encoding='utf-8') as dfd:
                for k, v in vin_img_num_dict.items():
                    dfd.write('{0}:{1}\n'.format(k, v))
            return vin_img_num_dict
        with open(dict_file, 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs0 = line.split(':')
                vin_code = arrs0[0]
                img_num = int(arrs0[1])
                vin_img_num_dict[vin_code] = img_num
        return vin_img_num_dict


    @staticmethod
    def get_import_vehicle_vin_set_img_num(vin_img_num_dict):
        num = 0
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
        for brand_obj in base_path.iterdir():
            for model_obj in brand_obj.iterdir():
                for year_obj in model_obj.iterdir():
                    for file_obj in year_obj.iterdir():
                        full_fn = str(file_obj)
                        if not file_obj.is_dir() and full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                            num += 1
                            WxsDsm.process_img_file_by_vin(vin_img_num_dict, '进口车', full_fn, num)

    @staticmethod
    def get_domestic_vehicle_vin_set_img_num(vin_img_num_dict):
        num = 0
        base_path = Path('/media/zjkj/work/guochanchezuowan-all')
        for block_obj in base_path.iterdir():
            for vin_obj in block_obj.iterdir():
                for file_obj in vin_obj.iterdir():
                    full_fn = str(file_obj)
                    if not file_obj.is_dir() and full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                        num += 1
                        WxsDsm.process_img_file_by_vin(vin_img_num_dict, '国产车', full_fn, num)

    @staticmethod
    def process_img_file_by_vin(vin_img_num_dict, img_type, full_fn, num):
        arrs0 = full_fn.split('/')
        img_file = arrs0[-1]
        arrs1 = img_file.split('_')
        arrs2 = arrs1[0].split('#')
        vin_code = arrs2[0]
        if vin_code in vin_img_num_dict:
            vin_img_num_dict[vin_code] += 1
        else:
            vin_img_num_dict[vin_code] = 1
        if num % 1000 == 0:
            print('处理{0}图片{1}个'.format(img_type, num))

    @staticmethod
    def get_wxs_empty_brands():
        '''
        找出无锡所Excel表格中图片数为零的品牌列表
        '''
        pass

    @staticmethod
    def get_wxs_brands():
        '''
        获取无锡所品牌列表
        '''
        brands = CBrand.get_wxs_brands()
        with open('./support/wxs_brands.csv', 'w+', encoding='utf-8') as bfd:
            for vo in brands:
                bfd.write('{0},{1},{2}\n'.format(vo['brand_id'], vo['brand_name'], vo['brand_code']))

    @staticmethod
    def get_brand_bm_bmy_of_samples():
        '''
        从samples.txt文件中统计出品牌数、车型数、年款数
        '''
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        brand_set = set()
        bm_set = set()
        bmy_set = set()
        num = 0
        #sample_file = './support/samples.txt'
        #sample_file = './support/raw_bid_train_ds.txt'
        sample_file = './support/raw_bid_train_ds.txt'
        with open(sample_file, 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs0 = line.split('*')
                bmy_id = int(arrs0[1]) + 1
                bmy_set.add(bmy_id)
                bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                brand_set.add(int(bmy_vo['brand_id']))
                bm_set.add(int(bmy_vo['model_id']))
                num += 1
                if num % 1000 == 0:
                    print('已经处理{0}条记录'.format(num))
        print('共有品牌{0}个，车型{1}个，年款{2}个'.format(len(brand_set), len(bm_set), len(bmy_set)))

    @staticmethod
    def bind_brand_head_bmy_head():
        '''
        实现先预测出品牌类别，然后从年款头中除该品牌对应的年款索引外的其他
        类别全部清零，将年款头的内容输出作为输出
        '''
        # 列出年款文件./support/cambricon_vehicle_label.txt内容
        id_bmy_dict = {}
        row = 0
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                id_bmy_dict[row] = line
                row += 1
        # 获取品牌列表
        brand_idx_bmys = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                brand_idx = int(arrs0[0])
                brand_name = arrs0[1]
                brand_idx_bmys[brand_idx] = []
                for k, v in id_bmy_dict.items():
                    if v.startswith(brand_name):
                        brand_idx_bmys[brand_idx].append(k)
        # 生成Python代码LoadModel.py中的结构代码
        print('{')
        for k, v in brand_idx_bmys.items():
            print('{0}:{1},'.format(k, v))
        print('}')
        # 生成C++后处理中的Vector初始化代码
        print('############################')
        print('{')
        for k, v in brand_idx_bmys.items():
            row_str = '{'
            first_item = True
            for vi in v:
                if first_item:
                    row_str += '{0}'.format(vi)
                    first_item = False
                else:
                    row_str += ',{0}'.format(vi)
            row_str += '},'
            print(row_str)
        print('}')

    @staticmethod
    def get_diff_wxs_tds_brands():
        '''
        求出无锡所测试集品牌与当前涉及的171个品牌的不同
        '''
        brand_name_idx_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                brand_idx = int(arrs0[0])
                brand_name = arrs0[1]
                brand_name_idx_dict[brand_name] = brand_idx
        wxs_brand_id_brand_name_dict = CBrand.get_wxs_brand_id_brand_name_dict()
        num_not_in_wxs = 0
        num_not_in_known = 0
        recs_num = 0
        brand_id_set = set()
        brand_id_set1 = set()
        brand_id_set2 = set()
        brand_id_ok_set = set()
        with open('./support/wxs_test_dataset_brands.csv', 'r', encoding='utf-8') as afd:
            for line in afd:
                line = line.strip()
                arrs0 = line.split(',')
                img_file = arrs0[0]
                brand_id = int(arrs0[1])
                brand_id_set.add(brand_id)
                if brand_id not in wxs_brand_id_brand_name_dict:
                    print('##### Error: {0};'.format(brand_id))
                    num_not_in_wxs += 1
                    brand_id_set1.add(brand_id)
                    continue
                brand_name = wxs_brand_id_brand_name_dict[brand_id]
                if brand_name not in brand_name_idx_dict:
                    print('########### 品牌不在已知范围：{0};'.format(brand_name))
                    num_not_in_known += 1
                    brand_id_set2.add(brand_id)
                    continue
                brand_idx = brand_name_idx_dict[brand_name]
                print('{0}*99999*{1};'.format(img_file, brand_idx))
                recs_num += 1
                brand_id_ok_set.add(brand_id)
        print('共有品牌：{0}个'.format(len(brand_id_set)))
        print('不在无锡所Excel中品牌有{0}个，共{1}条记录'.format(len(brand_id_set1), num_not_in_wxs))
        for bi in brand_id_set1:
            if bi == -1:
                brand_vo = {
                    'brand_id': -1,
                    'brand_name': '未知',
                    'source_type': -1
                }
            else:
                brand_vo = CBrand.get_brand_vo_by_id(bi)
            print('    {0}: {1}; {2};'.format(bi, brand_vo['brand_name'], brand_vo['source_type']))
        print('不在现有品牌列表中品牌有{0}个，共{1}条记录'.format(len(brand_id_set2), num_not_in_known))
        for bi in brand_id_set2:
            if bi == -1:
                brand_vo = {
                    'brand_id': -1,
                    'brand_name': '未知',
                    'source_type': -1
                }
            else:
                brand_vo = CBrand.get_brand_vo_by_id(bi)
            print('    {0}: {1}; {2};'.format(bi, brand_vo['brand_name'], brand_vo['source_type']))
        print('共有品牌{0}个，记录{1}条'.format(len(brand_id_ok_set), recs_num))

    @staticmethod
    def mark_error_img_in_wxs_tds():
        '''
        标出无锡所测试集中可能出错的图片
        '''
        base_path = Path('/media/zjkj/work/fgvc_dataset/wxs/fine')
        img_ok_set = set()
        for brand_obj in base_path.iterdir():
            for model_obj in brand_obj.iterdir():
                for year_obj in model_obj.iterdir():
                    for file_obj in year_obj.iterdir():
                        full_fn = str(file_obj)
                        if file_obj.is_file() and full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                            arrs0 = full_fn.split('/')
                            raw_img_file = arrs0[-1]
                            arrs1 = raw_img_file.split('_')
                            img_file = '{0}_{1}_{2}_{3}_{4}_{5}_{6}.jpg'.format(
                                arrs1[0], arrs1[1], arrs1[2],
                                arrs1[3], arrs1[4], arrs1[5],
                                arrs1[6]
                            )
                            img_ok_set.add(img_file)
                            print('add {0} to set'.format(img_file))
        with open('./support/wxs_test_dataset_brands_error.csv', 'w+', encoding='utf-8') as wfd:
            with open('./support/wxs_test_dataset_brands.csv', 'r', encoding='utf-8') as afd:
                for line in afd:
                    line = line.strip()
                    arrs0 = line.split(',')
                    full_fn = arrs0[0]
                    arrs1 = full_fn.split('/')
                    img_file = arrs1[-1]
                    state = 'error'
                    print('####    {0};'.format(img_file))
                    if img_file in img_ok_set:
                        state = 'ok'
                    brand_id = int(arrs0[1])
                    brand_notes = arrs0[2]
                    wfd.write('{0},{1},{2},{3}\n'.format(full_fn, brand_id, state, brand_notes))

    @staticmethod
    def generate_wxs_test_dataset():
        '''
        根据Csv文件生成测试数据集，其中年款值为一个不正确的值，因此年款精度为0，
        只测品牌精度
        '''
        tds_img_dict = {}
        base_path = Path('/media/zjkj/work/fgvc_dataset/wxs/fine')
        for brand_obj in base_path.iterdir():
            for model_obj in brand_obj.iterdir():
                for year_obj in model_obj.iterdir():
                    for file_obj in year_obj.iterdir():
                        full_fn = str(file_obj)
                        if file_obj.is_file() and full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                            arrs0 = full_fn.split('/')
                            raw_img_file = arrs0[-1]
                            arrs1 = raw_img_file.split('_')
                            img_file_key = '{0}_{1}_{2}_{3}_{4}_{5}_{6}.jpg'.format(
                                arrs1[0], arrs1[1], arrs1[2],
                                arrs1[3], arrs1[4], arrs1[5],
                                arrs1[6]
                            )
                            tds_img_dict[img_file_key] = full_fn
        # 生成现有171个品牌的品牌名称到索引号的字典
        brand_name_brand_idx_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                brand_idx = int(arrs0[0])
                brand_name = arrs0[1]
                brand_name_brand_idx_dict[brand_name] = brand_idx
        total = 0
        num_brand_in_wxs = 0
        num_brand_in_dcl = 0
        with open('./support/wxs_brands_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/wxs_test_dataset_brands_error.csv', 'r', encoding='utf-8') as tfd:
                wxs_brand_id_brand_name_dict = CBrand.get_wxs_brand_id_brand_name_dict()
                for line in tfd:
                    total += 1
                    line = line.strip()
                    arrs0 = line.split(',')
                    raw_img_file = arrs0[0]
                    arrs1 = raw_img_file.split('/')
                    img_file = arrs1[-1]
                    if img_file in tds_img_dict:
                        full_fn = tds_img_dict[img_file]
                    else:
                        full_fn = '/media/zjkj/work/品牌/{0}'.format(raw_img_file)
                    brand_id = int(arrs0[1])
                    if brand_id in wxs_brand_id_brand_name_dict:
                        brand_name = wxs_brand_id_brand_name_dict[brand_id]
                        num_brand_in_wxs += 1
                    else:
                        brand_name = '未知'
                    if brand_name in brand_name_brand_idx_dict:
                        brand_idx = brand_name_brand_idx_dict[brand_name]
                        if arrs0[2] == 'ok':
                            num_brand_in_dcl += 1
                            wfd.write('{0}*88888*{1}\n'.format(full_fn, brand_idx))
                    else:
                        brand_idx = 99999
                    print('{0}*88888*{1};'.format(full_fn, brand_idx))
                    #wfd.write('{0}*88888*{1}\n'.format(full_fn, brand_idx))
        print('品牌在无锡所Excel中的数量：{0}个，占{1}%'.format(num_brand_in_wxs, num_brand_in_wxs / total))
        print('在当前模型品牌列表中记录数为：{0}个，占{1}%'.format(num_brand_in_dcl, num_brand_in_dcl / total))

    @staticmethod
    def process_detect_jsons():
        '''
        将图片通过client1.8目录下的run.sh，发到服务器后，服务器会返回图像识别结果步骤：
        1. 遍历原始文件目录，形成一个文件名-全路径名的字典；
        2. 读取JSON文件，解析出原始文件名；
        3. 根据JSON文件中位置信息进行功图；
        4. 将切好的图直接缩放为224*244保存到scale1目录；
        5. 将切好的图按照长边缩放到224，短边0填充方式缩放到224*224，保存到scale2目录；
        6. 文件按车辆识别码目录进行组织；
        '''
        # 遍历原始目录得到文件名和全路径名的字典
        base_path = Path('/media/zjkj/work/品牌')
        img_file_full_fn_dict = {}
        WxsDsm.get_img_file_full_fn_dict(img_file_full_fn_dict, base_path)
        json_path = Path('/media/zjkj/work/yantao/w1/t001')
        num = 0
        bad_num = 0
        for json_obj in json_path.iterdir():
            json_file = str(json_obj)
            if not json_file.endswith('json'):
                continue
            arrs0 = json_file.split('/')
            jf = arrs0[-1]
            arrs1 = jf.split('_')
            img_file = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(
                arrs1[0], arrs1[1], arrs1[2], arrs1[3],
                arrs1[4], arrs1[5], arrs1[6]
            )
            img_full_fn = img_file_full_fn_dict[img_file]
            box_raw = WxsDsm.parse_detect_json(json_file)
            if box_raw is None:
                bad_num += 1
                continue
            arrs2 = box_raw.split(',')
            box = [int(arrs2[0]), int(arrs2[1]), int(arrs2[2]), int(arrs2[3])]
            crop_img = WxsDsm.crop_and_resize_img(img_full_fn, box)
            id_str = '{0:06d}'.format(num)
            folder1 = '/media/zjkj/work/yantao/zjkj/test_ds/{0}'.format(id_str[:2])
            if not os.path.exists(folder1):
                os.mkdir(folder1)
            folder2 = '{0}/{1}'.format(folder1, id_str[2:4])
            if not os.path.exists(folder2):
                os.mkdir(folder2)
            dst_file = '{0}/{1}'.format(folder2, img_file)
            num += 1
            cv2.imwrite(dst_file, crop_img)
        print('共处理{0}个文件，其中失败文件数为{1}个'.format(num + bad_num, bad_num))

    g_num = 0
    @staticmethod
    def get_img_file_full_fn_dict(img_file_full_fn_dict, base_path):
        for sub_obj in base_path.iterdir():
            if sub_obj.is_dir():
                WxsDsm.get_img_file_full_fn_dict(img_file_full_fn_dict, sub_obj)
            else:
                full_fn = str(sub_obj)
                if full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                    arrs0 = full_fn.split('/')
                    img_file = arrs0[-1]
                    if img_file in img_file_full_fn_dict:
                        print('### {0}有重名文件：'.format(img_file))
                        print('      第一个位置：{0};'.format(img_file_full_fn_dict[img_file]))
                        print('      第二个位置：{0};'.format(full_fn))
                    img_file_full_fn_dict[img_file] = full_fn
                    WxsDsm.g_num += 1

    @staticmethod
    def parse_detect_json(json_file):
        cllxfls = ['11', '12', '13', '14', '21', '22']
        with open(json_file, 'r', encoding='utf-8') as jfd:
            data = json.load(jfd)
        if len(data['VEH']) < 1:
            return None
        else:
            # 找到面积最大的检测框作为最终检测结果
            max_idx = -1
            max_area = 0
            for idx, veh in enumerate(data['VEH']):
                cllxfl = veh['CXTZ']['CLLXFL'][:2]
                if cllxfl in cllxfls:
                    box_str = veh['WZTZ']['CLWZ']
                    arrs_a = box_str.split(',')
                    x1, y1, w, h = int(arrs_a[0]), int(arrs_a[1]), int(arrs_a[2]), int(arrs_a[3])
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_idx = idx
            if max_idx < 0:
                return None
            else:
                return data['VEH'][max_idx]['WZTZ']['CLWZ']

    @staticmethod
    def process_es0901_jsons():
        '''
        解析9月1号测试错误图片车辆检测JSON文件
        '''
        #imgpath = '/media/zjkj/work/yantao/zjkj/es_crop/BH7161TMV_贵A5W20G_02_520000100860_520000104305010005.jpg'
        #imgpath = '/media/zjkj/work/yantao/zjkj/es_crop/ZN6460WAS_贵A2J97H_02_520000100685_520000104861443836.jpg'
        imgpath = '/media/zjkj/work/yantao/zjkj/es_crop/LZ6460Q9GE_赣A57S62_02_360000100650_360000200965450556.jpg'
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                img_obj = img.convert('RGB')
        print('img_obj: {0};'.format(img_obj))
        i_debug = 1
        if 1 == i_debug:
            return
        json_num = 0
        error_num = 0
        img_base_folder = '/media/zjkj/work/yantao/zjkj/es_images'
        crop_base_folder = '/media/zjkj/work/yantao/zjkj/es_crop'
        base_path = Path('/media/zjkj/work/yantao/zjkj/es_results')
        for file_obj in base_path.iterdir():
            full_fn = str(file_obj)
            json_num += 1
            clwz = WxsDsm.parse_detect_json(full_fn)
            if clwz is not None:
                arrs_a = full_fn.split('/')
                raw_json_fn = arrs_a[-1]
                img_file = raw_json_fn[:-7]
                img_full_fn = '{0}/{1}'.format(img_base_folder, img_file)
                print('{0} clwz: {1};'.format(img_full_fn, clwz))
                arrs2 = clwz.split(',')
                box = [int(arrs2[0]), int(arrs2[1]), int(arrs2[2]), int(arrs2[3])]
                if box[0] < 0:
                    box[0] = 0
                if box[1] < 0:
                    box[1] = 0
                crop_img = WxsDsm.crop_and_resize_img(img_full_fn, box)
                cv2.imwrite('{0}/{1}'.format(crop_base_folder, img_file), crop_img)
            else:
                print('    Error: {0};'.format(full_fn))
                error_num += 1
        print('总文件数：{0}; 错误数量：{1};'.format(json_num, error_num))

    @staticmethod
    def crop_and_resize_img(img_file, box, size=(224, 224), mode=1):
        if mode == 1:
            return WxsDsm.crop_and_resize_no_aspect(img_file, box, size)
        else:
            return WxsDsm.crop_and_resize_keep_aspect(img_file, box, size)

    @staticmethod
    def crop_and_resize_no_aspect(img_file, box, size=(224, 224), mode=1):
        org_img = cv2.imread(img_file)
        crop_img = org_img[
            box[1] : box[1] + box[3],
            box[0] : box[0] + box[2]
        ]
        '''
        plt.subplot(1, 3, 1)
        plt.title('org_img: {0}*{1}'.format(org_img.shape[0], org_img.shape[1]))
        plt.imshow(org_img)
        plt.subplot(1, 3, 2)
        plt.title('img: {0}*{1}'.format(crop_img.shape[0], crop_img.shape[1]))
        plt.imshow(crop_img)
        resized_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_LINEAR)
        plt.subplot(1, 3, 3)
        plt.title('resized')
        plt.imshow(resized_img)
        plt.show()
        '''
        return crop_img

    @staticmethod
    def crop_and_resize_keep_aspect(img_file, box, size=(224, 224), mode=1):
        pass

    @staticmethod
    def merge_zhangcan_csv():
        '''
        将张灿5634条记录的表格与6125张表格进行合并，将张灿的结果写入6125个表格
        相应的行中
        '''
        zc_dict = {}
        with open('./logs/zhangcan.csv', 'r', encoding='utf-8') as zfd:
            for line in zfd:
                line = line.strip()
                arrs0 = line.split(',')
                arrs1 = arrs0[0].split('/')
                img_file = arrs1[-1]
                zc_dict[img_file] = {
                    'brand_id': int(arrs0[1]),
                    'status': arrs0[2],
                    'notes': arrs0[3]
                }
        with open('./support/zhangcan_new.csv', 'w+', encoding='utf-8') as wfd:
            with open('./logs/wxs_tds_images.csv', 'r', encoding='utf-8') as ifd:
                for line in ifd:
                    line = line.strip()
                    arrs0 = line.split(',')
                    full_fn = arrs0[0]
                    arrs1 = full_fn.split('/')
                    img_file = arrs1[-1]
                    brand_id = -1
                    status = 'unknown'
                    notes = ''
                    if img_file in zc_dict:
                        brand_id = zc_dict[img_file]['brand_id']
                        status = zc_dict[img_file]['status']
                        notes = zc_dict[img_file]['notes']
                    wfd.write('{0},{1},{2},{3}\n'.format(full_fn, brand_id, status, notes))

    @staticmethod
    def generate_cut_img_test_ds():
        '''
        将所里测试集中文件替换为切图后文件
        '''
        test_ds_file = './datasets/CUB_200_2011/anno/wxs_brands_ds.txt'
        org_test_ds_dict = {}
        with open(test_ds_file, 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split('*')
                arrs_b = arrs_a[0].split('/')
                raw_img_file = arrs_b[-1]
                arrs_c = raw_img_file.split('_')
                img_file = '{0}_{1}_{2}_{3}_{4}_{5}_{6}.jpg'.format(
                    arrs_c[0], arrs_c[1], arrs_c[2],
                    arrs_c[3], arrs_c[4], arrs_c[5], arrs_c[6]
                )
                org_test_ds_dict[img_file] = {
                    'bmy_id': int(arrs_a[1]),
                    'brand_id': int(arrs_a[2])
                }
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds')
        img_file_full_fn_dict = {}
        WxsDsm.get_cut_test_ds_img_file_full_fn_dict(img_file_full_fn_dict, base_path)
        with open('./datasets/CUB_200_2011/anno/cut_wxs_brands_ds.txt', 'w+', encoding='utf-8') as wfd:
            for k, v in org_test_ds_dict.items():
                full_fn = img_file_full_fn_dict[k]
                wfd.write('{0}*{1}*{2}\n'.format(full_fn, v['bmy_id'], v['brand_id']))

    @staticmethod
    def get_cut_test_ds_img_file_full_fn_dict(img_file_full_fn_dict, base_path):
        num = 0
        for sub_obj in base_path.iterdir():
            if sub_obj.is_dir():
                WxsDsm.get_cut_test_ds_img_file_full_fn_dict(img_file_full_fn_dict, sub_obj)
            else:
                full_fn = str(sub_obj)
                arrs_a = full_fn.split('/')
                img_file = arrs_a[-1]
                img_file_full_fn_dict[img_file] = full_fn
                num += 1

    @staticmethod
    def cut_dataset_imgs():
        '''
        将训练集或随机抽取测试集图片拷贝到单独目录下，便于调用切图软件
        '''
        # ds_folder = '/media/zjkj/work/yantao/zjkj/train_ds_raw'
        ds_folder = '/media/zjkj/work/yantao/zjkj/work/random_tds_raw'
        # 将训练集图片拷贝到一个单独文件夹下
        num = 0
        with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split('*')
                img_full_fn = arrs_a[0]
                arrs_b = img_full_fn.split('/')
                img_file = arrs_b[-1]
                arrs_c = img_file.split('_')
                arrs_d = arrs_c[0].split('#')
                vin_code = arrs_d[0]
                dst_folder = '{0}/{1}'.format(ds_folder, vin_code)
                if not os.path.exists(dst_folder):
                    os.mkdir(dst_folder)
                dst_file = '{0}/{1}/{2}'.format(ds_folder, vin_code, img_file)
                shutil.copy(img_full_fn, dst_file)
                num += 1
                if num % 1000 == 0:
                    print('已经拷贝{0}条记录'.format(num))

    @staticmethod
    def find_diff_of_193_183():
        '''
        获取193万张原始图片和183万张检测图片之间，没有处理的图片列表，
        供后续查找原因
        '''
        # 取出183万图片set
        json_num = 0
        base_path = Path('/media/zjkj/work/yantao/zjkj/t003')
        detected_img_set = set()
        for sub_obj in base_path.iterdir():
            full_fn = str(sub_obj)
            if sub_obj.is_file() and full_fn.endswith(('json')):
                arrs_a = full_fn.split('/')
                json_file = arrs_a[-1]
                arrs_b = json_file.split('_')
                img_file = '{0}_{1}_{2}_{3}_{4}'.format(
                    arrs_b[0], arrs_b[1], arrs_b[2], 
                    arrs_b[3], arrs_b[4]
                )
                detected_img_set.add(img_file)
                json_num += 1
            if json_num % 100 == 0:
                print('已经处理{0}张图片...'.format(json_num))
        print('json_num={0};'.format(json_num))
        # 依次检查193万张，如果不包括在183集合中，将其记录下来
        ts_num = 0
        missing_files = []
        ts_set = set()
        with open('./datasets/CUB_200_2011/anno/bid_brand_train_ds.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                ts_num += 1
                ts_set.add(img_file)
                if img_file not in detected_img_set:
                    print('未包括文件：{0};'.format(full_fn))
                    missing_files.append(full_fn)
        print('缺失文件数为：{0}个；共{1}个文件；真实数量为{2}个;'.format(len(missing_files), ts_num, len(ts_set)))
        with open('./support/missing_files.txt', 'w+', encoding='utf-8') as mfd:
            for fn in missing_files:
                mfd.write('{0}\n'.format(fn))

    @staticmethod
    def get_img_file_full_fn_dict_from_ds_file(ds_file):
        num = 0
        # 遍历原始目录得到文件名和全路径名的字典
        img_file_full_fn_dict = {}
        #ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds.txt'
        with open(ds_file, 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                img_file_full_fn_dict[img_file] = full_fn
                num += 1
                if num % 10 == 0:
                    print('处理{0}条数据集文件记录'.format(num))
        print('生成图片文件名和全路径文件名字典')
        return img_file_full_fn_dict

    @staticmethod
    def get_cut_json_files(json_path):
        #json_path = Path('/media/zjkj/work/yantao/zjkj/work/random_tds_result')
        print('step 2')
        json_files = []
        jf_num = 0
        print('step 3')
        for json_obj in json_path.iterdir():
            json_file = str(json_obj)
            if not json_file.endswith('json'):
                continue
            json_files.append(json_file)
            jf_num += 1
            if jf_num % 100 == 0:
                print('处理完成{0}个json文件'.format(jf_num))
        return json_files

    @staticmethod
    def generate_crop_cv_img_thread(imgs_queue, img_file_full_fn_dict, json_files, finished_imgs):
        print('启动切图线程...')
        num = 0
        bad_num = 0
        bad_img_files = []
        for json_file in json_files:
            arrs0 = json_file.split('/')
            jf = arrs0[-1]
            arrs1 = jf.split('_')
            num += 1
            if num % 100 == 0:
                print('@@@@@@@@@@@@@@@     切图完成{0}个文件...'.format(num))
            img_file = '{0}_{1}_{2}_{3}_{4}'.format(
                arrs1[0], arrs1[1], arrs1[2], arrs1[3],
                arrs1[4]
            )
            if img_file in finished_imgs:
                continue
            img_full_fn = img_file_full_fn_dict[img_file]
            box_raw = WxsDsm.parse_detect_json(json_file)
            if box_raw is None:
                bad_num += 1
                bad_img_files.append(img_file)
                continue
            arrs2 = box_raw.split(',')
            box = [int(arrs2[0]), int(arrs2[1]), int(arrs2[2]), int(arrs2[3])]
            crop_img = WxsDsm.crop_and_resize_img(img_full_fn, box)
            imgs_queue.put({
                'img_file': img_file,
                'crop_img': crop_img
            })
        imgs_queue.put({
            'img_file': 'end',
            'crop_img': None
        })
        print('共处理{0}个文件，其中失败文件数为{1}个'.format(num + bad_num, bad_num))
        bad_imgs_txt = '/home/zjkj/client1.8/work/bad_images.txt'
        #bad_imgs_txt = './support/random_tds_bad_imgs.txt'
        with open(bad_imgs_txt, 'w+', encoding='utf-8') as bfd:
            for bi in bad_img_files:
                bfd.write('{0}\n'.format(bi))




    @staticmethod
    def save_crop_cv_img_thread(imgs_queue):
        print('启动保存图片文件线程...')
        num = 0
        while True:
            img_obj = imgs_queue.get()
            if img_obj['img_file'] == 'end':
                break
            img_file = img_obj['img_file']
            crop_img = img_obj['crop_img']
            arrs_a = img_file.split('_')
            arrs_b = arrs_a[0].split('#')
            vin_code = arrs_b[0]
            folder1 = '/home/zjkj/client1.8/work/cutted_images/{0}'.format(vin_code)
            #folder1 = '/media/zjkj/work/yantao/zjkj/work/random_tds/{0}'.format(vin_code)
            if not os.path.exists(folder1):
                os.mkdir(folder1)
            dst_file = '{0}/{1}'.format(folder1, img_file)
            num += 1
            cv2.imwrite(dst_file, crop_img)
            if num % 100 == 0:
                print('##########    保存完成{0}个...'.format(num))

    @staticmethod
    def process_training_ds_detect_jsons():
        '''
        将图片通过client1.8目录下的run.sh，发到服务器后，服务器会返回图像识别结果步骤：
        1. 遍历原始文件目录，形成一个文件名-全路径名的字典；
        2. 读取JSON文件，解析出原始文件名；
        3. 根据JSON文件中位置信息进行功图；
        4. 将切好的图直接缩放为224*244保存到scale1目录；
        5. 将切好的图按照长边缩放到224，短边0填充方式缩放到224*224，保存到scale2目录；
        6. 文件按车辆识别码目录进行组织；
        '''
        print('process_training_ds_detect_jsons')
        ds_file = './datasets/CUB_200_2011/anno/bid_brand_train_ds.txt'
        img_file_full_fn_dict = WxsDsm.get_img_file_full_fn_dict_from_ds_file(ds_file)
        print('从数据集文件获取图片文件名和全路径文件名字典')
        '''
        finished_imgs = WxsDsm.get_cut_finished_imgs()
        with open('./support/finished_imgs.txt', 'w+', encoding='utf-8') as ffd:
            for fi in finished_imgs:
                ffd.write('{0}\n'.format(fi))
        '''
        finished_imgs = set()
        with open('./support/finished_imgs.txt', 'r', encoding='utf-8') as ffd:
            for line in ffd:
                line = line.strip()
                finished_imgs.add(line)
        print('求出已经完成切图的文件列表 v0.0.1')
        json_path = Path('/media/zjkj/work/yantao/zjkj/t003')
        json_files = [] #WxsDsm.get_cut_json_files(json_path)
        with open('./support/detect_json_files.txt', 'r', encoding='utf-8') as jfd:
            for line in jfd:
                line = line.strip()
                json_files.append(line)
        print('求出所有车辆检测json文件列表')
        # 采用多线程方式运行
        imgs_queue = Queue(2000)
        # 启动切图线程
        cut_img_thd = threading.Thread(target=WxsDsm.generate_crop_cv_img_thread, args=(imgs_queue, img_file_full_fn_dict, json_files, finished_imgs))
        cut_img_thd.start()
        # 启动保存图片线程
        save_img_thd = threading.Thread(target=WxsDsm.save_crop_cv_img_thread, args=(imgs_queue,))
        save_img_thd.start()
        # 等待线程结束
        cut_img_thd.join()
        save_img_thd.join()
        print('^_^ The End! ^_^')




    @staticmethod
    def copy_detect_bad_image_files():
        '''
        将检测失败的图片文件拷贝到指定目录下，供耀辉改进其检测算法
        '''
        # 形成图片文件名和全咱路径文件名字典
        img_file_full_fn_dict = {}
        # ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds.txt'
        ds_file = './datasets/CUB_200_2011/anno/bid_brand_train_ds.txt'
        with open(ds_file, 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                img_file_full_fn_dict[img_file] = full_fn
        print('生成文件名到全路径文件名字典')
        # 拷贝文件
        # bad_imgs_txt = './support/random_tds_bad_imgs.txt'
        bad_imgs_txt = './support/train_bad_imgs.txt'
        # dst_folder = '/media/zjkj/work/yantao/zjkj/work/random_tds_bad_images'
        dst_folder = '/media/zjkj/work/yantao/zjkj/work/train_ds_bad_images'
        with open(bad_imgs_txt, 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                img_file = line
                full_fn = img_file_full_fn_dict[img_file]
                arrs_a = img_file.split('_')
                arrs_b = arrs_a[0].split('#')
                vin_code = arrs_b[0]
                dst_folder_vin_code = '{0}/{1}'.format(dst_folder, vin_code)
                if not os.path.exists(dst_folder_vin_code):
                    os.mkdir(dst_folder_vin_code)
                dst_file = '{0}/{1}'.format(dst_folder_vin_code, img_file)
                shutil.copy(full_fn, dst_file)

    @staticmethod
    def generate_cutted_dataset():
        '''
        生成由切过的图组成的数据集
        '''
        # 由数据集文件生成图片文件名和全路径名的字典
        # ds_file = './datasets/CUB_200_2011/anno/bid_brand_train_ds.txt'
        # ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds.txt'
        ds_file = './datasets/CUB_200_2011/anno/wxs_brands_ds.txt'
        img_file_sample_dict = WxsDsm.get_img_file_sample_dict_from_ds_file(ds_file)
        # base_path = Path('/media/zjkj/work/yantao/zjkj/train_ds')
        # base_path = Path('/media/zjkj/work/yantao/zjkj/work/random_tds')
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds')
        # cutted_ds_file = './datasets/CUB_200_2011/anno/train_ds_cut_v1.txt'
        # cutted_ds_file = './datasets/CUB_200_2011/anno/random_tds_v1.txt'
        cutted_ds_file = './datasets/CUB_200_2011/anno/wxs_brands_cut_ds.txt'
        num = 0
        with open(cutted_ds_file, 'w+', encoding='utf-8') as cfd:
            for vin_code_obj in base_path.iterdir():
                for img_file_obj in vin_code_obj.iterdir():
                    full_fn = str(img_file_obj)
                    arrs_a = full_fn.split('/')
                    img_file = arrs_a[-1]
                    print(img_file)
                    if img_file in img_file_sample_dict:
                        sample = img_file_sample_dict[img_file]
                        cfd.write('{0}*{1}*{2}\n'.format(full_fn, sample['bmy_id'], sample['brand_id']))
                        num += 1
                        if num % 100 == 0:
                            print('处理完成{0}我记录'.format(num))

    @staticmethod
    def generate_wxs_tds_cutted_dataset():
        '''
        生成由所里品牌测试切过的图组成的数据集
        '''
        # 由数据集文件生成图片文件名和全路径名的字典
        ds_file = './datasets/CUB_200_2011/anno/wxs_brands_ds.txt'
        img_file_sample_dict = WxsDsm.get_img_file_sample_dict_from_ds_file(ds_file)
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds')
        cutted_ds_file = './datasets/CUB_200_2011/anno/wxs_brands_cut_ds.txt'
        num = 0
        with open(cutted_ds_file, 'w+', encoding='utf-8') as cfd:
            for folder1_obj in base_path.iterdir():
                for folder2_obj in folder1_obj.iterdir():
                    for img_file_obj in folder2_obj.iterdir():
                        full_fn = str(img_file_obj)
                        arrs_a = full_fn.split('/')
                        raw_img_file = arrs_a[-1]
                        arrs_b = raw_img_file.split('_')
                        img_file = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_0.jpg'.format(
                            arrs_b[0], arrs_b[1], arrs_b[2], arrs_b[3],
                            arrs_b[4], arrs_b[5], arrs_b[6][:-4]
                        )
                        print(img_file)
                        if img_file in img_file_sample_dict:
                            sample = img_file_sample_dict[img_file]
                            cfd.write('{0}*{1}*{2}\n'.format(full_fn, sample['bmy_id'], sample['brand_id']))
                            num += 1
                            if num % 100 == 0:
                                print('处理完成{0}我记录'.format(num))

    @staticmethod
    def get_img_file_sample_dict_from_ds_file(ds_file):
        '''
        从数据集文件中得到图片文件名到全路径文件的字典
        '''
        # 形成图片文件名和全咱路径文件名字典
        img_file_sample_dict = {}
        with open(ds_file, 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                img_file_sample_dict[img_file] = {
                    'full_fn': full_fn,
                    'bmy_id': int(arrs_a[1]),
                    'brand_id': int(arrs_a[2])
                }
        print('生成文件名到全路径文件名字典')
        return img_file_sample_dict

    @staticmethod
    def get_cut_finished_imgs():
        # 遍历最终目录求出已完成切图的图片文件名列表
        num = 0
        finished_path = Path('/media/zjkj/work/yantao/zjkj/train_ds')
        finished_imgs = []
        for vc_obj in finished_path.iterdir():
            for file_obj in vc_obj.iterdir():
                full_fn = str(file_obj)
                arrs_a = full_fn.split('/')
                img_file = arrs_a[-1]
                finished_imgs.append(img_file)
                num += 1
                if num % 100 == 0:
                    print('统计完{0}个文件'.format(num))
        return finished_imgs

    @staticmethod
    def crop_image_demo():
        crop_ratio = 0.15
        org_img = cv2.imread('/media/zjkj/work/fgvc_dataset/raw/讴歌/rdx/2015/19UTB585#19UTB585_粤YUA739_02_440200100861_440200202556241574.jpg')
        img_w, img_h = org_img.shape[0], org_img.shape[1]
        img = org_img[
            int(crop_ratio*img_w):int((1-crop_ratio)*img_w), 
            int(crop_ratio*img_h):int((1-crop_ratio)*img_h)
        ]
        plt.subplot(1, 2, 1)
        plt.title('org_img: {0}*{1}'.format(img_w, img_h))
        plt.imshow(org_img)
        plt.subplot(1, 2, 2)
        plt.title('img: {0}*{1}'.format(img.shape[0], img.shape[1]))
        plt.imshow(img)
        plt.show()

    @staticmethod
    def check_wxs_tds_anno():
        '''
        检查无锡所测试集人工标注年款正确性，取出文件名中品牌车型年款，
        从年款编号求出品牌车型年款，再从DCL模型预测出一个品牌车型年款，
        如果三个不一致，则进行人工审核
        '''
        #with open('./datasets/CUB_200_2011/anno/wxs_')
        pass

    @staticmethod
    def generate_int8_quant_imgs_by_brand():
        '''
        通过随机采样生成int8量化需要的图片，以品牌为控制单位：
        当品牌下文件小于4个时，全部取；
        当大于4个时，随机从中取其中4张
        '''
        threshold_num = 5
        brand_idx_imgs_dict = {}
        with open('./datasets/CUB_200_2011/anno/train_ds_cut_v1.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                brand_idx = int(arrs_a[-1])
                if brand_idx not in brand_idx_imgs_dict:
                    brand_idx_imgs_dict[brand_idx] = [full_fn]
                else:
                    brand_idx_imgs_dict[brand_idx].append(full_fn)
        int8_imgs = []
        for k, v in brand_idx_imgs_dict.items():
            img_num = len(v)
            if img_num < threshold_num:
                for img_full_fn in v:
                    int8_imgs.append(img_full_fn)
            else:
                # 随机排序
                idx_list = [x for x in range(len(v))]
                random.shuffle(idx_list)
                for i in range(threshold_num):
                    int8_imgs.append(v[idx_list[i]])
        for img_full_fn in int8_imgs:
            print('### {0};'.format(img_full_fn))
            arrs_a = img_full_fn.split('/')
            img_file = arrs_a[-1]
            shutil.copy(img_full_fn, '/media/zjkj/work/yantao/zjkj/int8_images/{0}'.format(img_file))
        print('共有{0}张图片用于int8量化'.format(len(int8_imgs)))

    @staticmethod
    def get_brand_images_main():
        '''
        获取指定品牌在训练数据集和所里测试数据集中的图片，便于对比品牌预测
        错误图片的原因
        '''
        brand_idx = 137 # 威麟
        dst_folder = '/media/zjkj/work/yantao/zjkj/debug/b_135_137'
        ds_file = './datasets/CUB_200_2011/anno/train_ds_cut_v1.txt'
        WxsDsm.get_brand_images(brand_idx, dst_folder, ds_file)
        #
        ds_file = './datasets/CUB_200_2011/anno/wxs_brands_cut_ds.txt'
        WxsDsm.get_brand_images(brand_idx, dst_folder, ds_file)
        #
        brand_idx = 135
        ds_file = './datasets/CUB_200_2011/anno/train_ds_cut_v1.txt'
        WxsDsm.get_brand_images(brand_idx, dst_folder, ds_file)
        #
        ds_file = './datasets/CUB_200_2011/anno/wxs_brands_cut_ds.txt'
        WxsDsm.get_brand_images(brand_idx, dst_folder, ds_file)

    @staticmethod
    def get_brand_images(brand_idx, dst_folder, ds_file):
        with open(ds_file, 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split('*')
                pd_brand_idx = int(arrs_a[-1])
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                if pd_brand_idx == brand_idx:
                    shutil.copy(full_fn, '{0}/{1}/{2}'.format(
                        dst_folder, pd_brand_idx, img_file
                    ))

    
    @staticmethod
    def get_bmy_id_bm_vo_dict():
        '''
        获取bmy_id（年款头输出）与车型值对象的字典
        '''
        #bmy_id_model_vo_dict = CBmy.get_bmy_id_bm_vo_dict()
        #np.save('./support/bmy_bm.npy', bmy_id_model_vo_dict)
        return np.load('./support/bmy_bm.npy', allow_pickle=True).item()

    @staticmethod
    def get_bmy_sim_org_dict():
        bmy_sim_org_dict = {}
        with open('./support/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs_a = line.split(':')
                sim_bmy_id = int(arrs_a[0])
                org_bmy_id = int(arrs_a[1])
                bmy_sim_org_dict[sim_bmy_id] = org_bmy_id
        return bmy_sim_org_dict

    @staticmethod
    def integrate_wxs_tds_bmy():
        '''
        将无锡所测试集年款信息添加到训练集中进行训练
        '''
        num = 0
        img_file_full_fn_dict = {}
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds')
        for sub1_obj in base_path.iterdir():
            for sub2_obj in sub1_obj.iterdir():
                for file_obj in sub2_obj.iterdir():
                    full_fn = str(file_obj)
                    arrs_a = full_fn.split('/')
                    img_file = arrs_a[-1]
                    img_file_full_fn_dict[img_file] = full_fn
                    num += 1
        print('共有{0}个文件'.format(num))
        img_file_bmy_id_dict = {}
        with open('./support/wxs_tds_bmy.csv', 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split(',')
                arrs_b = arrs_a[0].split('/')
                img_file = arrs_b[-1]
                if arrs_a[1] != '':
                    org_bmy_id = int(arrs_a[1])
                else:
                    org_bmy_id = -1
                img_file_bmy_id_dict[img_file] = org_bmy_id
        missed_num = 0
        with open('./support/wxs_tds_v2.txt', 'w+', encoding='utf-8') as wfd:
            for k, v in img_file_full_fn_dict.items():
                img_file = k
                full_fn = v
                if img_file in img_file_bmy_id_dict and img_file_bmy_id_dict[img_file] >= 0:
                    wfd.write('{0}*{1}\n'.format(full_fn, img_file_bmy_id_dict[img_file]))
                else:
                    missed_num += 1
        print('未标注样本总数为：{0}个'.format(missed_num))

    @staticmethod
    def generate_cut_wxs_tds_merged_raw_ds():
        '''
        生成由切图过图像和所里切图过测试集合并在一起的原始数据（bmyId为原始值）
        '''
        # 读入图片文件名与年款编号的对应关系字典
        img_file_2_org_bmy_id_dict = {}
        with open('./support/raw_bid_train_ds.txt', 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs_a = line.split('*')
                org_bmy_id = int(arrs_a[1])
                arrs_b = arrs_a[0].split('/')
                img_file = arrs_b[-1]
                img_file_2_org_bmy_id_dict[img_file] = org_bmy_id
        print('读入图片文件名与年款编号（-1）字典...')
        # 读入train_ds目录下切过的图，由图片文件名与年款编号的对应关系字典求出
        # 年款编号
        samples = []
        base_path = Path('/media/zjkj/work/yantao/zjkj/train_ds')
        for vc_obj in base_path.iterdir():
            for file_obj in vc_obj.iterdir():
                full_fn = str(file_obj)
                arrs_a = full_fn.split('/')
                img_file = arrs_a[-1]
                org_bmy_id = img_file_2_org_bmy_id_dict[img_file]
                samples.append({
                    'img_full_fn': full_fn,
                    'org_bmy_id': org_bmy_id
                })
        print('读入切过图的训练集图片文件，并生成原始训练集...')
        # 读入无锡所切图过的测试原始数据集
        with open('./support/wxs_tds_v2.txt', 'r', encoding='utf-8') as afd:
            for line in afd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                org_bmy_id = arrs_a[1]
                samples.append({
                    'img_full_fn': full_fn,
                    'org_bmy_id': org_bmy_id
                })
        print('将无锡所测试集加入训练集')
        # 将所有内容写入到原始数据集文件中
        with open('./support/raw_bid_train_ds_v001.txt', 'w+', encoding='utf-8') as wfd:
            for sample in samples:
                wfd.write('{0}*{1}\n'.format(sample['img_full_fn'], sample['org_bmy_id']))

    @staticmethod
    def get_to_anno_wxs_tds():
        '''
        找出无锡所测试集中需要标注年款的记录
        '''
        # 找出已经标注过年款的图片文件名
        img_file_has_bmy_set = set()
        with open('./support/wxs_tds_bmy.csv', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                arrs_a = line.split(',')
                rel_img_file = arrs_a[0]
                arrs_b = rel_img_file.split('/')
                img_file = arrs_b[-1]
                if arrs_a[1] != '':
                    img_file_has_bmy_set.add(img_file)
        print('已经标注：{0}条;'.format(len(img_file_has_bmy_set)))
        # 找出无锡所测试集中文件名与全路径文件名的字典
        to_anno_images = []
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds')
        for sub1_obj in base_path.iterdir():
            for sub2_obj in sub1_obj.iterdir():
                for file_obj in sub2_obj.iterdir():
                    full_fn = str(file_obj)
                    arrs_a = full_fn.split('/')
                    img_file = arrs_a[-1]
                    if img_file not in img_file_has_bmy_set:
                        to_anno_images.append(full_fn)
        print('共有{0}条需要标注'.format(len(to_anno_images)))
        # 拷贝图片文件
        dst_folder = '/media/zjkj/work/yantao/w1/to_anno/images'
        for full_fn in to_anno_images:
            arrs_a = full_fn.split('/')
            img_file = arrs_a[-1]
            shutil.copy(full_fn, '{0}/{1}'.format(dst_folder, img_file))

    num1 = 0
    @staticmethod
    def get_bmy_example_images():
        '''
        获取无锡所品牌车型年款Excel中每个年款5张示例图片，并以品牌车型年款
        目录结构进行组织
        '''
        vin_code_2_images_dict = {}
        def process_vin_image(file_obj):
            global num
            full_fn = str(file_obj)
            if file_obj.is_file() and full_fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                arrs_a = full_fn.split('/')
                arrs_b = arrs_a[-1].split('_')
                arrs_c = arrs_b[0].split('#')
                vin_code = arrs_c[0]
                if vin_code not in vin_code_2_images_dict:
                    vin_code_2_images_dict[vin_code] = [full_fn]
                else:
                    vin_code_2_images_dict[vin_code].append(full_fn)
                WxsDsm.num1 += 1
                if WxsDsm.num1 % 1000 == 0:
                    print('已经处理完成{0}个文件'.format(WxsDsm.num1))
        # 获取VIN与图片列表关系字典
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
        for brand_obj in base_path.iterdir():
            for bm_obj in brand_obj.iterdir():
                for bmy_obj in bm_obj.iterdir():
                    for file_obj in bmy_obj.iterdir():
                        process_vin_image(file_obj)
        for vc_obj in Path('/media/zjkj/work/guochanchezuowan-all').iterdir():
            for sub_obj in vc_obj.iterdir():
                for file_obj in sub_obj.iterdir():
                    process_vin_image(file_obj)
        print('生成VIN与图片列表关系字典 v0.0.1')
        with open('./support/vin_code_2_images_dict.txt', 'wb') as vcfd:
            pickle.dump(vin_code_2_images_dict, vcfd)
        # 获取无锡所品牌车型年款列表
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        vin_code_vos = CBmy.get_wxs_vins()
        base_folder = '/media/zjkj/work/yantao/zjkj/bmy_example'
        for vo in vin_code_vos:
            vin_code = vo['vin_code']
            bmy_id = int(vo['bmy_id'])
            bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
            # 创建目录结构
            arrs_a = bmy_vo['bmy_name'].split('-')
            brand_name = arrs_a[0]
            brand_folder = '{0}/{1}'.format(base_folder, brand_name)
            if not os.path.exists(brand_folder):
                os.mkdir(brand_folder)
            model_name = arrs_a[1]
            model_folder = '{0}/{1}'.format(brand_folder, model_name)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            year_name = arrs_a[2]
            year_folder = '{0}/{1}'.format(model_folder, year_name)
            if not os.path.exists(year_folder):
                os.mkdir(year_folder)
            # 通过两个目录fgvc_dataset/raw和guochanchezuowan-all找到图片文件列表
            # 随机抽取5张图片拷贝到该目录下
            if vin_code in vin_code_2_images_dict:
                data = list(range(len(vin_code_2_images_dict[vin_code])))
                random.shuffle(data)
                selected_idxs = data[:10]
                for idx in selected_idxs:
                    img_full_fn = vin_code_2_images_dict[vin_code][idx]
                    arrs_e = img_full_fn.split('/')
                    img_file = arrs_e[-1]
                    shutil.copy(img_full_fn, '{0}/{1}'.format(year_folder, img_file))
                print('拷贝：{0}: {1};'.format(vin_code, bmy_vo['bmy_name']))
            else:
                print('#### Error vin_code: {0};'.format(vin_code))
        print('共有{0}条记录'.format(len(vin_code_vos)))

    @staticmethod
    def rectify_raw_bid_train_ds():
        '''
        修改数据集中人工标注错误，主要是无锡所测试集中有405条记录需要修改，
        其他记录原样复制
        '''
        rectify_dict = {}
        with open('./logs/wxs_tds_rectify_data.txt', 'r', encoding='utf-8') as fd1:
            for line in fd1:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                rectify_dict[full_fn] = arrs_a[1]
        with open('./support/raw_bid_train_ds_v001.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/raw_bid_train_ds_v001_bk.txt', 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    if full_fn in rectify_dict:
                        wfd.write('{0}*{1}\n'.format(full_fn, rectify_dict[full_fn]))
                    else:
                        wfd.write('{0}\n'.format(line))

    @staticmethod
    def process_error_sample_for_zhangcan():
        '''
        将模型预测错误的样本整理为图片文件名, 标注年款, 预测年款 格式，发
        给张灿，由其进行处理。如果模型预测错误，则标注模型错误，如果标注
        错误则写上新年款编号 2020-08-16
        '''
        bid_brands_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs_a = line.split(':')
                brand_idx = int(arrs_a[0])
                brand_name = arrs_a[1]
                bid_brands_dict[brand_idx] = brand_name
        dst_folder = '../work/dcl20200828/images'
        with open('../work/dcl20200828/error_samples.txt', 'w+', encoding='utf-8') as wfd:
            with open('./logs/top1_error_samples.txt', 'r', encoding='utf-8') as efd:
                for line in efd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = arrs_a[0].split('/')
                    img_file = arrs_b[-1]
                    shutil.copy(full_fn, '{0}/{1}'.format(dst_folder, img_file))
                    anno_bmy_id = int(arrs_a[1])
                    dcl_bmy_id = int(arrs_a[2])
                    wfd.write('{0},{1},{2},\n'.format(
                        img_file, 
                        bid_brands_dict[anno_bmy_id], 
                        bid_brands_dict[dcl_bmy_id]
                    ))
        # 求出全路径文件名和样本字符串字典
        full_fn_to_sample_dict = {}
        with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds_082701.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                full_fn_to_sample_dict[full_fn] = line
        with open('../work/dcl20200828/samples_to_duplicate.txt', 'w+', encoding='utf-8') as wfd:
            with open('./logs/top1_error_samples.txt', 'r', encoding='utf-8') as efd:
                for line in efd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    sample_line = full_fn_to_sample_dict[full_fn]
                    wfd.write('{0}\n'.format(sample_line))

    @staticmethod
    def process_error_sample_20200817():
        '''
        处理20200817经过修正后的分类错误的样本数据
        '''
        # 读取品牌名称到品牌索引号列表
        brand_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs_a = line.split(':')
                brand_dict[arrs_a[1]] = arrs_a[0]
        # 读取org_bmy_id到sim_bmy_id对应表
        bmy_org_sim_dict = {}
        with open('./support/bmy_org_sim_dict.txt', 'r', encoding='utf-8') as osfd:
            for line in osfd:
                line = line.strip()
                arrs_a = line.split(':')
                org_bmy_id = int(arrs_a[0])
                sim_bmy_id = int(arrs_a[1])
                bmy_org_sim_dict[org_bmy_id] = sim_bmy_id
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        es_dict = {}
        with open('/media/zjkj/work/yantao/zjkj/doc/d20200817/error_samples.txt', 'r', encoding='utf-8') as efd:
            for line in efd:
                line = line.strip()
                arrs_a = line.split(',')
                img_file = arrs_a[0]
                anno_bn = arrs_a[1]
                dcl_bn = arrs_a[2]
                arrs_b = arrs_a[3].split('-')
                bmy_name = arrs_b[0]
                org_bmy_id = int(arrs_b[-1])
                es_dict[img_file] = org_bmy_id
                bmy_id = int(arrs_b[-1]) + 1
                bmy_name_dst = bmy_id_bmy_name_dict[bmy_id]
                print('{0}, {1}, {2}, {3}, {4};'.format(
                    img_file, anno_bn, dcl_bn, bmy_name, bmy_name_dst
                ))
        with open('./support/raw_bid_train_ds_v001.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/raw_bid_train_ds_v000.txt', 'r', encoding='utf-8') as ofd:
                for line in ofd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    img_file = arrs_b[-1]
                    if img_file in es_dict:
                        bmy_id = es_dict[img_file]
                        wfd.write('{0}*{1}\n'.format(full_fn, bmy_id))
                    else:
                        wfd.write('{0}\n'.format(line))

    @staticmethod
    def calculate_img_similarity_main():
        '''
        计算两张图片的相似度，通过获取两张图片ReID特征向量，然后计算二者的余弦
        距离作为相似度
        '''
        img1 = 'E:/work/tcv/a001/白#02_陕DBD888_001_奥迪_A6L_2005-2011_610500200969342729.jpg'
        #img2 = 'E:/work/tcv/a001/ext_白#02_川R171C1_012_'\
        #    '比亚迪_L3_2010-2013_610500200969346136.jpg'
        img2 = 'E:/work/tcv/a001/白#02_川R171C1_012_比亚迪_L3_2010-2013_610500200969346136.jpg'
        WxsDsm.calculate_img_similarity(img1, img2)
        img3 = 'E:/work/tcv/a001/白#02_豫A75DD7_001_奥迪_A6L_2012-2014_610500200969339543.jpg'
        WxsDsm.calculate_img_similarity(img1, img3)
        img4 = 'E:/work/tcv/a001/白#06_粤T20R21_001_奥迪_A6L_2012-2014_610500200969339816.jpg'
        WxsDsm.calculate_img_similarity(img1, img4)

    @staticmethod
    def calculate_img_similarity(img1, img2):
        img1_fv = WxsDsm.get_img_reid_feature_vector(img1)
        img2_fv = WxsDsm.get_img_reid_feature_vector(img2)
        print('img1: {0}; {1};'.format(type(img1_fv), img1_fv.shape))
        print('img2: {0}; {1};'.format(type(img2_fv), img2_fv.shape))
        img1_norm = np.linalg.norm(img1_fv, axis=0, keepdims=True)
        img2_norm = np.linalg.norm(img2_fv, axis=0, keepdims=True)
        similiarity = np.dot(img1_fv, img2_fv.T)/(img1_norm * img2_norm) 
        print('similarity: {0};'.format(similiarity[0]))

    @staticmethod
    def get_img_reid_feature_vector(full_fn):
        url = 'http://192.168.2.17:2222/vehicle/function/recognition'
        print('url: {0};'.format(url))
        data = {'TPLX': 1, 'GCXH': 123131318}
        files = {'TPWJ': (full_fn, open(full_fn, 'rb'))}
        resp = requests.post(url, files=files, data = data)
        json_obj = json.loads(resp.text)
        vehs = json_obj['VEH']
        raw = []
        for veh in vehs:
            cllxfl = veh['CXTZ']['CLLXFL']
            if cllxfl in ['11', '12', '13', '14', '21', '22']:
                print('detected...')
                vals = veh['CLTZXL'].split(',')
                for val in vals:
                    raw.append(float(val))
        return np.array(raw)

    @staticmethod
    def move_detect_json_to_folder():
        '''
        定期从车辆检测结果目录下读取检测的Json文件，将其存储到每100个文件一个目录的格式，
        并且保存原有的文件名。
        '''
        print('移动车辆检测Json文件')
        # 生成1万个文件
        '''
        for i in range(10085):
            with open('/media/zjkj/work/yantao/temp/t001/a_{0}.txt'.format(i), 'w+', encoding='utf-8') as wfd:
                wfd.write('内容：{0:03d}个消息\n'.format(i))
        '''
        file_exts = ('txt', 'jpg', 'json')
        base_path = Path('/media/zjkj/work/yantao/zjkj/detect_raw_1')
        num = 0
        dst_folder = '/media/zjkj/work/yantao/zjkj/detect_1'

        while True:
            print('移动文件线程启动...')
            for jo in base_path.iterdir():
                full_fn = str(jo)
                if jo.is_file() and full_fn.endswith(file_exts):
                    print('移动：{0}文件'.format(full_fn))
                    arrs_a = full_fn.split('/')
                    json_file = arrs_a[-1]
                    raw_str = '{0:0>8d}'.format(num)
                    print('raw_str: {0};'.format(raw_str))
                    df1 = '{0}/{1}'.format(dst_folder, raw_str[:2])
                    if not os.path.exists(df1):
                        os.mkdir(df1)
                    df2 = '{0}/{1}'.format(df1, raw_str[2:4])
                    if not os.path.exists(df2):
                        os.mkdir(df2)
                    df3 = '{0}/{1}'.format(df2, raw_str[4:6])
                    if not os.path.exists(df3):
                        os.mkdir(df3)
                    '''
                    df = '{0}/{1}'.format(df3, raw_str[6:])
                    if not os.path.exists(df):
                        os.mkdir(df)
                    '''
                    dst_file = '{0}/{1}'.format(df3, json_file)
                    shutil.move(full_fn, dst_file)
                    num += 1
            time.sleep(1)

    @staticmethod
    def divide_samples():
        '''
        将1400万样本分割为10万一个单位，并保存为独立的文件: 
        samples_001.txt ~ sample_140.txt
        '''
        file_id = 0
        idx = 1
        files = []
        with open('./support/samples_v001.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                #bmy_id = int(arrs_a[1])
                files.append(full_fn)
                if idx % 100000 == 0:
                    # 保存入一个新文件
                    sample_fn = './support/raw_sample_{0:05d}.txt'.format(file_id)
                    with open(sample_fn, 'w+', encoding='utf-8') as nfd:
                        for ff in files:
                            nfd.write('{0}\n'.format(ff))
                    files = []
                    file_id += 1
                if idx % 1000 == 0:
                    print('读取{0}个样本'.format(idx))
                idx += 1
        sample_fn = './support/raw_sample_{0:05d}.txt'.format(file_id)
        with open(sample_fn, 'w+', encoding='utf-8') as nfd:
            for ff in files:
                nfd.write('{0}\n'.format(ff))

    @staticmethod
    def copy_sample_image_files_main():
        '''
        将10万个样本文件中的图片拷贝到client1.8/work/sample_files目录下，并按每100个
        文件一个目录组织
        '''
        sample_file = '/home/zjkj/client1.8/work/sample_list/raw_sample_00000.txt'
        WxsDsm.copy_sample_image_files(sample_file)

    @staticmethod
    def copy_sample_image_files(sample_file):
        dst_base_folder = '/home/zjkj/client1.8/work/sample_files'
        num = 0
        with open(sample_file, 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                fn = arrs_b[-1]
                raw_str = '{0:0>6d}'.format(num)
                sub1 = '{0}/{1}'.format(dst_base_folder, raw_str[:2])
                if not os.path.exists(sub1):
                    os.mkdir(sub1)
                dst_folder = '{0}/{1}'.format(sub1, raw_str[2:4])
                if not os.path.exists(dst_folder):
                    os.mkdir(dst_folder)
                dst_file = '{0}/{1}'.format(dst_folder, fn)
                shutil.copy(full_fn, dst_file)
                num += 1
                if num % 100 == 0:
                    print('完成{0}个文件拷贝'.format(num))

    
    @staticmethod
    def process_seg_ds_detect_jsons():
        '''
        将图片通过client1.8目录下的run.sh，发到服务器后，服务器会返回图像识别结果步骤：
        1. 遍历原始文件目录，形成一个文件名-全路径名的字典；
        2. 读取JSON文件，解析出原始文件名；
        3. 根据JSON文件中位置信息进行功图；
        4. 将切好的图直接缩放为224*244保存到scale1目录；
        5. 将切好的图按照长边缩放到224，短边0填充方式缩放到224*224，保存到scale2目录；
        6. 文件按车辆识别码目录进行组织；
        '''
        print('process_training_ds_detect_jsons')
        finished_imgs = set()
        ds_file = '/home/zjkj/client1.8/work/sample_list/raw_sample_00000.txt'
        img_file_full_fn_dict = {} #WxsDsm.get_img_file_full_fn_dict_from_ds_file(ds_file)
        with open(ds_file, 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs_a = line.split('/')
                img_file = arrs_a[-1]
                img_file_full_fn_dict[img_file] = line
        print('从数据集文件获取图片文件名和全路径文件名字典')
        json_path = Path('/home/zjkj/client1.8/work/detect_results')
        json_files = [] #WxsDsm.get_cut_json_files(json_path)
        for jf in json_path.iterdir():
            json_files.append(str(jf))
        print('求出所有车辆检测json文件列表')
        # 采用多线程方式运行
        imgs_queue = Queue(2000)
        # 启动切图线程
        cut_img_thd = threading.Thread(target=WxsDsm.generate_crop_cv_img_thread, args=(imgs_queue, img_file_full_fn_dict, json_files, finished_imgs))
        cut_img_thd.start()
        # 启动保存图片线程
        save_img_thd = threading.Thread(target=WxsDsm.save_crop_cv_img_thread, args=(imgs_queue,))
        save_img_thd.start()
        # 等待线程结束
        cut_img_thd.join()
        save_img_thd.join()
        print('^_^ The End! ^_^')

    @staticmethod
    def copy_cut_bad_images_pair():
        '''
        将切图错误的图片原图和切过的图拷贝到同一目录下
        '''
        bad_image_files = []
        with open('/home/zjkj/client1.8/work/bad_image_files_yt.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                if line != '':
                    bad_image_files.append(line)
        bad_images_folder = '/home/zjkj/client1.8/work/bad_images'
        for bif in bad_image_files:
            print('deal with {0};'.format(bif))
            org_base_path = Path('/home/zjkj/client1.8/work/sample_files')
            for sub1 in org_base_path.iterdir():
                for sub2 in sub1.iterdir():
                    for file_obj in sub2.iterdir():
                        full_fn = str(file_obj)
                        arrs_a = full_fn.split('/')
                        img_file = arrs_a[-1]
                        if bif == img_file:
                            shutil.copy(full_fn, '{0}/{1}_org.jpg'.format(bad_images_folder, bif[:-4]))
            cut_base_path = Path('/home/zjkj/client1.8/work/cutted_images')
            for vf in cut_base_path.iterdir():
                for file_obj in vf.iterdir():
                    full_fn = str(file_obj)
                    arrs_a = full_fn.split('/')
                    img_file = arrs_a[-1]
                    if bif == img_file:
                        shutil.copy(full_fn, '{0}/{1}_cut.jpg'.format(bad_images_folder, bif[:-4]))

    @staticmethod
    def generate_txt_by_wxs_tds_ok_images():
        '''
        将无锡所测试集中标注出年款的图片文件名和年款编号写进文件文件中
        '''
        base_path = Path('E:/work/tcv/temp/wlok')
        for img_obj in base_path.iterdir():
            full_fn = str(img_obj)
            arrs_a = full_fn.split('-')
            prefix = arrs_a[0]
            arrs_b = arrs_a[0].split('\\')
            bmy_code = arrs_b[-1]
            img_file = full_fn[len(prefix)+1:]
            print('{0}: {1};'.format(bmy_code, img_file))

    @staticmethod
    def correct_augment_raw_ds():
        '''
        修改原始数据集中标注错误的样本，增加出错样本个数，生成新的数据集
        '''
        bmy_code_to_bmy_id_dict = CBmy.get_bmy_code_to_bmy_id_dict()
        img_file_to_result_dict = {}
        # 生成修改字典
        with open('./logs/correct_samples.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[1]
                bmy_code = arrs_a[-1]
                org_bmy_id = bmy_code_to_bmy_id_dict['{0} '.format(bmy_code)] - 1
                print('{0}*{1};'.format(img_file, org_bmy_id))
                img_file_to_result_dict[img_file] = org_bmy_id
        num = 0
        with open('./support/raw_bid_train_ds_v002.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/raw_bid_train_ds_v001.txt', 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    img_file = arrs_b[-1]
                    if img_file in img_file_to_result_dict:
                        print('### {0}*{1};'.format(full_fn, img_file_to_result_dict[img_file]))
                        wfd.write('{0}*{1}\n'.format(full_fn, img_file_to_result_dict[img_file]))
                    else:
                        wfd.write('{0}\n'.format(line))
                    num += 1
                    if num %1000 == 0:
                        print('已经处理完成{0}条记录'.format(num))

    @staticmethod
    def duplicate_miss_samples():
        '''
        复制指定份数必错的样本，希望能够不再出错
        '''
        duplicate_copys = 10
        img_file_set = set()
        with open('./logs/miss_samples.txt', 'r', encoding='utf-8') as mfd:
            for line in mfd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[1]
                img_file_set.add(img_file)
        num = 1
        with open('./support/bid_brand_train_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/bid_brand_train_ds_temp.txt', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    img_file = arrs_b[-1]
                    if img_file in img_file_set:
                        for i in range(duplicate_copys):
                            wfd.write('{0}\n'.format(line))
                    else:
                        wfd.write('{0}\n'.format(line))
                    num += 1
                    if num % 1000 == 0:
                        print('已经处理完成{0}条记录'.format(num))

    @staticmethod
    def wxs_tds_to_image_brand():
        '''
        求出无锡所测试集图片与品牌名称对应关系
        '''
        brand_idx_to_brand_name_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs_a = line.split(':')
                brand_idx = int(arrs_a[0])
                brand_name = arrs_a[1]
                brand_idx_to_brand_name_dict[brand_idx] = brand_name
        with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds_082801.txt', 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                brand_idx = int(arrs_a[2])
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                brand_name = brand_idx_to_brand_name_dict[brand_idx]
                print('{0}*{1};'.format(img_file, brand_name))
    
    @staticmethod
    def exp001():
        sim_dict = {}
        with open('./support/cpp_bp.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                arrs_a = line.split(':')
                idx = int(arrs_a[0])
                content = arrs_a[1].replace('[', '{').replace(']', '}')
                sim_dict[idx] = content
        with open('./support/cpp_bp_vec.txt', 'w+', encoding='utf-8') as wfd:
            for idx in range(1500+1):
                if idx in sim_dict:
                    wfd.write('{0}\n'.format(sim_dict[idx]))
                else:
                    wfd.write('{0},\n')


    @staticmethod
    def generate_samples_wxs0901():
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        bmy_id_bmy_vo = CBmy.get_bmy_id_bmy_vo_dict()
        img_base_folder = '../datasets/es_crop'
        base_path = Path(img_base_folder)
        error_num = 0
        b_num = 0
        total = 0
        with open('../work/wxs_es0901_ds.txt', 'w+', encoding='utf-8') as wfd:
            for file_obj in base_path.iterdir():
                full_fn = str(file_obj)
                arrs_a = full_fn.split('/')
                img_file = arrs_a[-1]
                arrs_b = img_file.split('_')
                arrs_c = arrs_b[0].split('#')
                vin_code = arrs_c[0]
                #print('{0}: {1};'.format(img_file, vin_code))
                if vin_code in vin_code_bmy_id_dict:
                    bmy_id = int(vin_code_bmy_id_dict[vin_code])
                    bmy_vo = bmy_id_bmy_vo[bmy_id]
                    wfd.write('{0}*{1}\n'.format('{0}/{1}'.format(img_base_folder, img_file), bmy_id))
                    total += 1
                    if bmy_vo['bmy_code'].startswith('b'):
                        print('{0}: {1}-{2};'.format(img_file, bmy_id, bmy_vo['bmy_code']))
                        b_num += 1
                else:
                    bmy_id = -1
                    error_num += 1
                    print('### Error: {0} = {1};'.format(img_file, vin_code))
        print('total: {0}; brand_error: {1}; error num: {2};'.format(total, b_num, error_num))
        
    @staticmethod
    def correct_vin_bmy_codes_error():
        '''
        将9月1日测试错误的车辆识别码由我们的编号转换为所里的编号
        '''
        vc_to_bc = {}
        with open('../work/dcl/bid_20200903.csv', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs_a = line.split(',')
                vin_code = arrs_a[-1]
                brand_code = arrs_a[1]
                bm_code = arrs_a[3]
                bmy_code = arrs_a[5]
                vc_to_bc[vin_code] = {
                    'brand_code': brand_code,
                    'bm_code': bm_code,
                    'bmy_code': bmy_code
                }
        print('获取品牌车型年款编码完成')
        # 获取需要修正的车辆识别码列表
        vin_codes = []
        with open('../work/dcl/f1.log', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                arrs_a = line.split('_')
                vin_code = arrs_a[0]
                vin_codes.append(vin_code)
        print('获取需要修改的车辆识别码列表完成')
        for vin_code in vin_codes:
            vo = vc_to_bc[vin_code]
            brand_code = vo['brand_code']
            bm_code = vo['bm_code']
            bmy_code = vo['bmy_code']
            print('修正：{0} => {1};'.format(vin_code, bmy_code))
            bmy_id, _ = CBmy.get_bmy_id_by_vin_code(vin_code)
            CBmy.update_bmy_codes(bmy_id, bmy_code, bm_code, brand_code)

    @staticmethod
    def change_hyphen_to_underscore_in_vin_code():
        '''
        将数据库t_vin中所有车辆识别码中的_换成-，因为其会与分隔符
        混淆
        '''
        old_vin_codes = CBmy.get_vin_id_codes()
        for vc in old_vin_codes:
            vin_id = int(vc['vin_id'])
            org_vin_code = vc['vin_code']
            vin_code = org_vin_code.replace('_', '-')
            CBmy.update_vin_code_by_vin_id(vin_id, vin_code)
            print('### 更新： {0}: {1} 《= {2};'.format(vin_id, vin_code, org_vin_code))
            
    @staticmethod
    def check_wxs0901_missing_vins():
        # 求出数据集中车辆识别码图片数量字典
        def process_img_file(vc_to_in, full_fn, num):
            arrs_a = full_fn.split('/')
            img_file = arrs_a[-1]
            arrs_b = img_file.split('_')
            arrs_c = arrs_b[0].split('#')
            vin_code = arrs_c[0]
            if vin_code in our_vc_to_in:
                our_vc_to_in[vin_code] += 1
            else:
                our_vc_to_in[vin_code] = 1
            num += 1
            if num % 1000 == 0:
                print('已经处理{0}条记录'.format(num))
            return num
        print('找出无锡所测试缺失车辆识别码...')
        def process_imported_vehicles(our_vc_to_in):
            base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
            num = 0
            for brand_obj in base_path.iterdir():
                for bm_obj in brand_obj.iterdir():
                    for bmy_obj in bm_obj.iterdir():
                        for file_obj in bmy_obj.iterdir():
                            full_fn = str(file_obj)
                            if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                                num = process_img_file(our_vc_to_in, full_fn, num)
        def process_domestic_vehicles(our_vc_to_in):
            base_path = Path('/media/zjkj/work/guochanchezuowan-all')
            num = 0
            for sub_obj in base_path.iterdir():
                for vc_obj in sub_obj.iterdir():
                    for file_obj in vc_obj.iterdir():
                        full_fn = str(file_obj)
                        if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                            num = process_img_file(our_vc_to_in, full_fn, num)
        def process_domestic_2n(our_vc_to_in):
            base_path = Path('/media/zjkj/work/guochanche_2n')
            num = 0
            for vf_obj in base_path.iterdir():
                for file_obj in vf_obj.iterdir():
                    full_fn = str(file_obj)
                    if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                        num = process_img_file(our_vc_to_in, full_fn, num)
        '''
        our_vc_to_in = {}
        process_imported_vehicles(our_vc_to_in)
        process_domestic_vehicles(our_vc_to_in)
        with open('./support/vc_img_num.txt', 'w+', encoding='utf-8') as ifd:
            for k, v in our_vc_to_in.items():
                ifd.write('{0}:{1}\n'.format(k, v))
        '''
        our_vc_to_in = {}
        process_domestic_2n(our_vc_to_in)
        for k, v in our_vc_to_in.items():
            print('### {0}:{1};'.format(k, v))
        i_debug = 1
        if 1 == i_debug:
            return
        with open('../work/dcl/vc_img_num.txt', 'r', encoding='utf-8') as ifd:
            for line in ifd:
                line = line.strip()
                arrs_a = line.split(':')
                vin_code = arrs_a[0]
                img_num = int(arrs_a[1])
                our_vc_to_in[vin_code] = img_num
        print('读取文件系统车辆识别号和图片文件数完成')
        vin_code_to_img_num = {}
        with open('./datasets/CUB_200_2011/anno/bid_brand_train_ds_090501.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                arrs_c = img_file.split('_')
                arrs_d = arrs_c[0].split('#')
                vin_code = arrs_d[0]
                if vin_code in vin_code_to_img_num:
                    vin_code_to_img_num[vin_code] += 1
                else:
                    vin_code_to_img_num[vin_code] = 1
        print('读取数据集车辆识别号和图片文件数完成')
        for key in vin_code_to_img_num.keys():
            if key not in our_vc_to_in:
                print('缺失车辆识别码：{0};'.format(key))


    @staticmethod
    def norm_files_folder():
        print('规整文件存储目录')
        saver = VdJsonSaver()
        saver.start()
        i_debug = 1
        if 1 == i_debug:
            return
        def create_tree_folder(parent_folder, child_folder):
            dst_folder = '{0}/d{1}'.format(parent_folder, child_folder)
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)
            return dst_folder
        def create_folder(base_folder, file_id):
            full_str = '{0:012d}'.format(file_id)
            folder1 = create_tree_folder(base_folder, full_str[:2])
            folder2 = create_tree_folder(folder1, full_str[2:4])
            folder3 = create_tree_folder(folder2, full_str[4:6])
            folder4 = create_tree_folder(folder3, full_str[6:8])
            folder5 = create_tree_folder(folder4, full_str[8:10])
            return folder5
        base_folder = '/media/zjkj/work/bdb_images'
        base_path = Path('/media/zjkj/work/g2ne')
        file_id = 0
        num = 0
        for sub0_obj in base_path.iterdir():
            for vc_obj in sub0_obj.iterdir():
                for file_obj in vc_obj.iterdir():
                    full_fn = str(file_obj)
                    if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                        dst_folder = create_folder(base_folder, file_id)
                        arrs_a = full_fn.split('/')
                        img_file = arrs_a[-1]
                        shutil.move(full_fn, '{0}/{1}'.format(dst_folder, img_file))
                        file_id += 1
                        num += 1
                        if num % 100 == 0:
                            print('移动{0}个文件'.format(num))
        with open('{0}/file_id.txt'.format(base_folder), 'w+', encoding='utf-8') as fd:
            ffd.write('{0}'.format(file_id))
            
            
    @staticmethod
    def refine_prev_dataset():
        vin_code_to_bmy_id = CBmy.get_wxs_vin_code_bmy_id_dict()
        with open('./support/new_bid_train_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/bid_train_ds.txt', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    img_file = arrs_b[-1]
                    arrs_c = img_file.split('_')
                    arrs_d = arrs_c[0].split('#')
                    vin_code = arrs_d[0]
                    print('{0}---{1}'.format(img_file, vin_code))
                    if vin_code in vin_code_to_bmy_id:
                        bmy_id = vin_code_to_bmy_id[vin_code]
                    else:
                        vin_had_bmy_id = False
                        for k, _ in vin_code_to_bmy_id.items():
                            if k.startswith(vin_code):
                                bmy_id = vin_code_to_bmy_id[k]
                                vin_had_bmy_id = True
                                break
                        if not vin_had_bmy_id:
                            bmy_id = -1
                            if vin_code != '白' and vin_code != '夜':
                                print('{0}\n'.format(vin_code))
                    if bmy_id > 0:
                        wfd.write('{0}*{1}\n'.format(full_fn, bmy_id - 1))
                
                
    @staticmethod
    def process_vd_jsons():
        vjm = VdJsonManager()
        vjm.start()
        
    @staticmethod
    def check_gcc2n_vin_codes():
        bid_vcs = BidVin.get_vin_codes()
        print('标书共{0}个车辆识别码'.format(len(bid_vcs)))
        gcc2_vcs = Gcc2nDm.get_gcc2n_vcs()
        print('gcc2n有{0}个车辆识别码'.format(len(gcc2_vcs)))
        intersection = bid_vcs & gcc2_vcs
        print('交集：{0};'.format(len(intersection)))
        
    @staticmethod
    def run_lmdb_demo():
        ImageLmdb.demo()
        
    @staticmethod
    def save_ds_imgs_to_lmdb():
        '''
        将所有数据集文件（通过遍历特定格式文件夹）保存到LMDB中
        '''
        # 读出并显示一个图片
        ImageLmdb.initialize_lmdb()
        img_full_fn = '/media/zjkj/work/yantao/zjkj/test_ds/00/20/白#02_陕AY13C9_033_福特_福克斯_2007_610500200969342197.jpg'
        img = ImageLmdb.get_image_multi(img_full_fn)
        plt.imshow(img)
        plt.show()
        ImageLmdb.destroy()
        i_debug = 1
        if 1 == i_debug:
            return
        ImageLmdb.initialize_lmdb()
        ImageLmdb.save_ds_imgs_to_lmdb()
        ImageLmdb.destroy()
        
    @staticmethod
    def get_files_in_subfolders_dict():
        base_folder = '/media/zjkj/work/fgvc_dataset/vdc0907/json_500'
        FileUtil.get_files_in_subfolders_dict(base_folder)
        
    @staticmethod
    def run_vd_cut_save():
        VdJsonManager.run_vd_cut_save()
        
    @staticmethod
    def run_vd_cut_save_on_es20200923():
        VdJsonManager.run_vd_cut_save_on_es20200923()
        
    @staticmethod
    def run_vd_cut_save_on_wxs_ds():
        VdJsonManager.run_vd_cut_save_on_wxs_ds()
        
    @staticmethod
    def delete_error_samples_main():
        src_ds_file = './datasets/CUB_200_2011/anno/bid_brand_train_ds_090601.txt'
        dst_ds_file = './datasets/CUB_200_2011/anno/bid_brand_train_ds_091001.txt'
        WxsDsm.delete_error_samples(src_ds_file, dst_ds_file)
        # test dataset
        src_ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds_090601.txt'
        dst_ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds_091001.txt'
        WxsDsm.delete_error_samples(src_ds_file, dst_ds_file)
        
    @staticmethod
    def delete_error_samples(src_ds_file, dst_ds_file):
        delete_files = set()
        with open('./support/delete_files.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split(',')
                img_file = arrs_a[0]
                delete_files.add(img_file)
        with open(dst_ds_file, 'w+', encoding='utf-8') as wfd:
            with open(src_ds_file, 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    img_file = arrs_b[-1]
                    if img_file not in delete_files:
                        wfd.write('{0}\n'.format(line))
                    else:
                        print('删除：{0};'.format(img_file))
                        
    @staticmethod
    def delete_wxs_error_20200919():
        '''
        从9月1日错误集中删除品牌错误和车尾的数据
        '''
        error_images = set()
        with open('./logs/wxs_error_20200919.txt', 'r', encoding='utf-8') as efd:
            for line in efd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[0]
                error_images.add(img_file)
        # 生成新的训练集
        def convert_ds_file(src_ds_file, dst_ds_file):
            with open(dst_ds_file, 'w+', encoding='utf-8') as w_trfd:
                with open(src_ds_file, 'r', encoding='utf-8') as r_trfd:
                    for line in r_trfd:
                        line = line.strip()
                        arrs_a = line.split('*')
                        full_fn = arrs_a[0]
                        arrs_b = full_fn.split('/')
                        img_file = arrs_b[-1]
                        if img_file not in error_images:
                            w_trfd.write('{0}\n'.format(line))
                        else:
                            print('去掉错误记录{0};'.format(img_file))
        src_ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds_091001.txt'
        dst_ds_file = './datasets/CUB_200_2011/anno/bid_brand_test_ds_20200919.txt'
        convert_ds_file(src_ds_file, dst_ds_file)
        
        
    @staticmethod
    def add_old_wxs_brand_ds():
        bmy_sim_org_dict = {}
        with open('./support/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
            for line in sofd:
                line = line.strip()
                arrs_a = line.split(':')
                sim_bmy_id = int(arrs_a[0])
                org_bmy_id = int(arrs_a[1])
                bmy_sim_org_dict[sim_bmy_id] = org_bmy_id
        with open('./support/raw_bid_train_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/bid_train_ds.txt', 'r', encoding='utf-8') as bfd:
                for line in bfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    sim_bmy_id = int(arrs_a[1])
                    if sim_bmy_id not in bmy_sim_org_dict:
                        print('Error!!!!!!!!!!!!!!!!!!!!!!!!! {0}; {1};'.format(sim_bmy_id, full_fn))
                    org_bmy_id = bmy_sim_org_dict[sim_bmy_id]
                    wfd.write('{0}*{1}\n'.format(full_fn, org_bmy_id))
        i_debug = 1
        if 1 == i_debug:
            return
        bmy_org_sim_dict = {}
        num = 0
        with open('./support/bmy_org_sim_dict.txt', 'r', encoding='utf-8') as osfd:
            for line in osfd:
                line = line.strip()
                arrs_a = line.split(':')
                org_bmy_id = int(arrs_a[0])
                sim_bmy_id = int(arrs_a[1])
                bmy_org_sim_dict[org_bmy_id] = sim_bmy_id
        print('读入原始年款编号与精简年款编号字典...')
        with open('./support/wxs_raw_ds_20200922.txt', 'r', encoding='utf-8') as xfd:
            for line in xfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                org_bmy_id = int(arrs_a[1])
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                arrs_c = img_file.split('_')
                brand_name = arrs_c[3]
                if org_bmy_id not in bmy_org_sim_dict:
                    print('img_file: {0}; bmy_id: {1};'.format(img_file, org_bmy_id))
                    num += 1
        print('error num: {0}'.format(num))
        '''
        cvl_dict = {}
        val = 0
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as lfd:
            for line in lfd:
                line = line.strip()
                arrs_a = line.split(',')
                key = '{0},{1},{2}'.format(arrs_a[0], arrs_a[1], arrs_a[2])
                cvl_dict[key] = val
                val += 1
        print('val={0};'.format(val))
        num = 0
        with open('./datasets/CUB_200_2011/anno/bid_brand_train_ds_090901.txt', 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs_a = line.split('*')
                full_fn = arrs_a[0]
                arrs_b = full_fn.split('/')
                img_file = arrs_b[-1]
                arrs_c = img_file.split('_')
                brand_name = '{0}牌'.format(arrs_c[3])
                model_name = arrs_c[4]
                year_raw = arrs_c[5]
                year_name = year_raw.replace('-', '_')
                key = '{0},{1},{2}'.format(brand_name, model_name, year_name)
                if key not in cvl_dict:
                    num += 1
        print('num={0};'.format(num))
        '''
        
    @staticmethod
    def generate_fds_samples():
        '''
        生成539万进口车和900万国产车图片形成的全量数据集样本集
        '''
        print('生成全量数据集样本集...')
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        oprr_num = 0
        with open('./support/samples_v002.txt', 'w+', encoding='utf-8') \
                        as sfd:
            with open('./support/error_vins.txt', 'w+', encoding='utf-8') \
                            as efd:
                # 进口车目录
                import_folder = '/media/ps/work/yantao/zjkj/raw_cutted'
                oprr_num = WxsDsm.process_image_folder(
                    import_folder, oprr_num, 
                    vin_code_bmy_id_dict, 
                    bmy_id_bmy_name_dict, brand_set,
                    sfd, efd
                )
                # 国产车已处理
                domestic_folder = '/media/ps/work/yantao/zjkj/i900m_cutted'
                oprr_num = WxsDsm.process_image_folder(
                    domestic_folder, oprr_num, 
                    vin_code_bmy_id_dict, 
                    bmy_id_bmy_name_dict, brand_set,
                    sfd, efd
                )
        print('已经处理品牌数：{0};'.format(len(brand_set)))
        
    
    @staticmethod
    def process_image_folder(folder_name, oprr_num, vin_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, sfd, efd):
        for root, dirs, files in os.walk(folder_name, topdown=False):
            for fn in files:
                if fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                    full_fn = '{0}/{1}'.format(root, fn)
                    oprr_num = WxsDsm.process_one_sample_image(
                        oprr_num, vin_bmy_id_dict, 
                        bmy_id_bmy_name_dict, brand_set, 
                        full_fn, sfd, efd
                    )
        return oprr_num

    @staticmethod
    def process_one_sample_image(oprr_num, vin_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, sub_file, sfd, efd):
        '''
        对每个图片进行处理（不包括所里原来的品牌测试集图片）
        '''
        arrs0 = sub_file.split('/')
        filename = arrs0[-1]
        arrs1 = filename.split('_')
        raw_vin_code = arrs1[0]
        arrs2 = raw_vin_code.split('#')
        vin_code = arrs2[0]
        if vin_code in vin_bmy_id_dict:
            bmy_id = vin_bmy_id_dict[vin_code]
        else:
            vin_had_bmy_id = False
            for k, _ in vin_bmy_id_dict.items():
                if k.startswith(vin_code):
                    bmy_id = vin_bmy_id_dict[k]
                    vin_had_bmy_id = True
                    break
            if not vin_had_bmy_id:
                bmy_id = -1
                if vin_code != '白' and vin_code != '夜':
                    efd.write('{0}\n'.format(vin_code))
        if bmy_id > 0:
            sfd.write('{0}*{1}\n'.format(sub_file, bmy_id - 1))
            bmy_name = bmy_id_bmy_name_dict[bmy_id]
            arrsn = bmy_name.split('-')
            brand_name = arrsn[0]
            brand_set.add(brand_name)
        oprr_num += 1
        if oprr_num % 1000 == 0:
            print('处理{0}条记录...'.format(oprr_num))
        return oprr_num
        
    @staticmethod
    def generate_raw_ds_esc20200923():
        '''
        生成2020-09-23无锡所测试错误图片切图后图片的原始数据集
        '''
        print('生成全量数据集样本集...')
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        oprr_num = 0
        with open('./support/raw_ds_esc20200923.txt', 'w+', encoding='utf-8') \
                        as sfd:
            with open('./support/esc_20200923_error_vins.txt', 'w+', encoding='utf-8') \
                            as efd:
                # 进口车目录
                import_folder = './support/ds_files/esc20200923'
                oprr_num = WxsDsm.process_image_folder(
                    import_folder, oprr_num, 
                    vin_code_bmy_id_dict, 
                    bmy_id_bmy_name_dict, brand_set,
                    sfd, efd
                )
        print('已经处理品牌数：{0};'.format(len(brand_set)))
        
    @staticmethod
    def get_vd_failed_images():
        '''
        找出无锡所9月23日品牌错误图片中未检测出图片列表
        '''
        vd_set = set()
        with open('./support/raw_ds_esc20200923.txt', 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs_a = line.split('*')
                rel_fn = arrs_a[0]
                arrs_b = rel_fn.split('/')
                img_file = arrs_b[-1]
                vd_set.add(img_file)
        failed_images = []
        with open('./support/es20200923_images.txt', 'r', encoding='utf-8') as efd:
            for line in efd:
                line = line.strip()
                arrs_a = line.split('/')
                img_file = arrs_a[-1]
                if img_file not in vd_set:
                    print('未检出文件{0};'.format(img_file))
                    failed_images.append(img_file)
        print('共有{0}张图片未检出'.format(len(failed_images)))
        with open('../vd_failed_images.txt', 'w+', encoding='utf-8') as wfd:
            for img in failed_images:
                wfd.write('{0}\n'.format(img))
                
    @staticmethod
    def copy_fine_random_images():
        '''
        将随机抽取的切图图片替换为质量更高的切图图片，即更新train_ds目录
        下的图片内容
        '''
        rds_set = set()
        with open('./support/rds_set.txt', 'r', encoding='utf-8') as fd1:
            for line in fd1:
                line = line.strip()
                rds_set.add(line)
        rds_dict = {}
        with open('./support/rds_dict.txt', 'r', encoding='utf-8') as fd2:
            for line in fd2:
                line = line.strip()
                arrs_a = line.split(':')
                rds_dict[arrs_a[0]] = arrs_a[1]
        fds_dict = {}
        with open('./support/fds_dict.txt', 'r', encoding='utf-8') as fd3:
            for line in fd3:
                line = line.strip()
                arrs_a = line.split(':')
                fds_dict[arrs_a[0]] = arrs_a[1]
        num = 0
        for img_file in rds_set:
            if img_file in fds_dict and img_file in rds_dict:
                src_file = fds_dict[img_file]
                dst_file = rds_dict[img_file]
                shutil.copy(src_file, dst_file)
                num += 1
                if num % 100 == 0:
                    print('拷贝{0}个图片'.format(num))
        i_debug = 1
        if 1 == i_debug:
            return        
        # 读出随机抽取train_ds目录下图片文件名和全路径文件名字典
        tnum = 0
        train_ds_images = set()
        train_ds_dict = {}
        folder_name = './support/ds_files/train_ds'
        def add_to_image_set(img_set, img_file):
            img_set.add(img_file)
        def nop_func(img_set, img_file):
            pass
        def process_folder(folder_name, ffn_dict, ffn_set, fn_func, num):
            for root, dirs, files in os.walk(folder_name, topdown=False):
                for fn in files:
                    if fn.endswith(('jpg', 'png', 'jpeg', 'bmp')):
                        full_fn = '{0}/{1}'.format(root, fn)
                        fn_func(ffn_set, fn)
                        ffn_dict[fn] = full_fn
                        num += 1
                        if num % 100 == 0:
                            print('处理随机训练集{0}张图片'.format(num))
            return num
        process_folder(folder_name, train_ds_dict, train_ds_images, add_to_image_set, tnum)
        print('随机测试集数量：{0};'.format(len(train_ds_images)))
        with open('./support/rds_set.txt', 'w+', encoding='utf-8') as wfd:
            for img_file in train_ds_images:
                wfd.write('{0}\n'.format(img_file))
        with open('./support/rds_dict.txt', 'w+', encoding='utf-8') as wfd:
            for k, v in train_ds_dict.items():
                wfd.write('{0}:{1}\n'.format(k, v))
        fds_dict = {}
        fds_num = 0
        raw_folder = '/media/ps/work/yantao/zjkj/raw_cutted'
        fds_num = process_folder(raw_folder, fds_dict, None, nop_func, fds_num)
        d900_folder = '/media/ps/work/yantao/zjkj/i900m_cutted'
        process_folder(d900_folder, fds_dict, None, nop_func, fds_num)
        wnum = 0
        with open('./support/fds_dict.txt', 'w+', encoding='utf-8') as wfd:
            for k, v in fds_dict.items():
                wfd.write('{0}:{1}\n'.format(k, v))
                wnum += 1
                if wnum % 100 == 0:
                    print('写入{0}条记录'.format(wnum))
                    
    @staticmethod
    def add_err_imgs_to_rds():
        print('将错误样本加入到训练集中去...')
        WxsDsm.add_err_imgs_to_rds1()
        i_debug = 1
        if 1 == i_debug:
            return
        bmy_org_sim_dict = {}
        with open('./support/bmy_org_sim_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs_a = line.split(':')
                bmy_org_sim_dict[int(arrs_a[0])] = int(arrs_a[1])
        sim_bmy_id = 0
        sim_dict = {}
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                arrs_a = line.split(',')
                sim_dict[sim_bmy_id] = arrs_a[0] # brand_name
                sim_bmy_id += 1
        brand_name_idx_dict = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as dfd:
            for line in dfd:
                line = line.strip()
                arrs_a = line.split(':')
                brand_name_idx_dict[arrs_a[1]] = int(arrs_a[0])
        with open('./support/esi_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/esi_samples.txt', 'r', encoding='utf-8') as sfd:
                for line in sfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    org_bmy_id = int(arrs_a[1])
                    print('org_bmy_id: {0};'.format(org_bmy_id))
                    if org_bmy_id in bmy_org_sim_dict:
                        sim_bmy_id = bmy_org_sim_dict[org_bmy_id]
                        brand_name = sim_dict[sim_bmy_id]
                        brand_idx = brand_name_idx_dict[brand_name]
                        img_file = arrs_a[0]
                        wfd.write('./support/ds_files/es_crop/{0}*{1}*{2}\n'.format(img_file, sim_bmy_id, brand_idx))
        
    @staticmethod
    def add_err_imgs_to_rds1():
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        oprr_num = 0
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        sfd = open('./support/esi_samples.txt', 'w+', encoding='utf-8')
        efd = open('./support/esi_errors.txt', 'w+', encoding='utf-8')
        with open('./support/error_images_wxs20200901.txt', 'r', encoding='utf-8') as afd:
            for line in afd:
                line = line.strip()
                # 获取车辆识别码
                WxsDsm.process_err_one_img_file(oprr_num, vin_code_bmy_id_dict,
                    bmy_id_bmy_name_dict, brand_set, line, sfd, efd
                )
        sfd.close()
        efd.close()
        
        
    
    @staticmethod
    def process_err_one_img_file(oprr_num, vin_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, sub_file, sfd, efd):
        arrs0 = sub_file.split('/')
        filename = arrs0[-1]
        arrs1 = filename.split('_')
        raw_vin_code = arrs1[0]
        arrs2 = raw_vin_code.split('#')
        vin_code = arrs2[0]
        print('vin_code={0};'.format(vin_code))
        if vin_code in vin_bmy_id_dict:
            bmy_id = vin_bmy_id_dict[vin_code]
        else:
            vin_had_bmy_id = False
            for k, _ in vin_bmy_id_dict.items():
                if k.startswith(vin_code):
                    bmy_id = vin_bmy_id_dict[k]
                    vin_had_bmy_id = True
                    break
            if not vin_had_bmy_id:
                bmy_id = -1
                if vin_code != '白' and vin_code != '夜':
                    efd.write('{0}\n'.format(vin_code))
        if bmy_id > 0:
            print('       {0}*{1};'.format(sub_file, bmy_id - 1))
            sfd.write('{0}*{1}\n'.format(sub_file, bmy_id - 1))
            bmy_name = bmy_id_bmy_name_dict[bmy_id]
            arrsn = bmy_name.split('-')
            brand_name = arrsn[0]
            brand_set.add(brand_name)
        oprr_num += 1
        if oprr_num % 1000 == 0:
            print('处理{0}条记录...'.format(oprr_num))
        return oprr_num
        
    @staticmethod
    def purify_full_ds_main():
        print('清理全量数据集...')
        #WxsDsm.pfdm_train_ds()
        #WxsDsm.pfdm_test_ds()
        #WxsDsm.generate_fds_cambricon_labels()
        WxsDsm.bind_fds_brand_head_bmy_head()
    

    @staticmethod
    def generate_fds_cambricon_labels():
        '''
        由新的bmy_id求出老的bmy_id，然后求出品牌车型年款并用逗号分隔，生成一个txt文件
        '''
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        with open('./support/cambricon_vehicle_label.txt', 'w+', encoding='utf-8') as fd:
            for bmy_id, bmy_vo in bmy_id_bmy_vo_dict.items():
                bmy_name = bmy_vo['bmy_name']
                arrs0 = bmy_name.split('-')
                brand_name = arrs0[0]
                model_name = arrs0[1]
                year_name = arrs0[2]
                fd.write('{0},{1},{2},{3},{4},{5}\n'.format(
                    brand_name, model_name, year_name,
                    bmy_vo['brand_code'],
                    bmy_vo['model_code'],
                    bmy_vo['bmy_code']
                ))
        
    @staticmethod
    def pfdm_train_ds():
        # 获取错误样本列表
        es_set = set()
        num1 = 0
        with open('./support/esi_samples.txt', 'r', encoding='utf-8') as efd:
            for line in efd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[0]
                es_set.add(img_file)
                num1 += 1
        print('读出错误文件列表：{0}个，文件数：{1}个'.format(len(es_set), num1))
        # 获取bmy_id对应的brand_id
        bmy_id_2_bmy_vo = CBmy.get_bmy_id_2_bmy_vo()
        # 生成数据集
        num = 0
        with open('./support/fds_train_ds.txt', 'w+', encoding='utf-8') as ffd:
            with open('./support/raw_bid_train_ds.txt', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    fn = arrs_b[-1]
                    bmy_id = int(arrs_a[1]) + 1
                    if fn not in es_set and bmy_id in bmy_id_2_bmy_vo:
                        bmy_vo = bmy_id_2_bmy_vo[bmy_id]
                        brand_id = bmy_vo['brand_id']
                        ffd.write('{0}*{1}*{2}\n'.format(full_fn, bmy_id, brand_id))
                    num += 1
                    if num % 100 == 0:
                        print('处理完成{0}条记录'.format(num))
                        
                        
    @staticmethod
    def pfdm_test_ds():
        #WxsDsm.pre_pdfm_test_ds()
        # 获取bmy_id对应的brand_id
        bmy_id_2_bmy_vo = CBmy.get_bmy_id_2_bmy_vo()
        # 生成数据集
        num = 0
        with open('./support/fds_test_ds.txt', 'w+', encoding='utf-8') as ffd:
            with open('./support/raw_fds_test_ds.txt', 'r', encoding='utf-8') as tfd:
                for line in tfd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    full_fn = arrs_a[0]
                    arrs_b = full_fn.split('/')
                    fn = arrs_b[-1]
                    bmy_id = int(arrs_a[1]) + 1
                    if bmy_id in bmy_id_2_bmy_vo:
                        bmy_vo = bmy_id_2_bmy_vo[bmy_id]
                        brand_id = bmy_vo['brand_id']
                        ffd.write('{0}*{1}*{2}\n'.format(full_fn, bmy_id, brand_id))
                    else:
                        print('Error: {0};'.format(full_fn))
                    num += 1
                    if num % 100 == 0:
                        print('处理完成{0}条记录'.format(num))
                        
    
    @staticmethod
    def pre_pdfm_test_ds():
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        brand_set = set()
        oprr_num = 0
        with open('./logs/conflicts.txt', 'w+', encoding='utf-8') \
                        as WxsDsm.g_cfd:
            with open('./support/raw_fds_test_ds.txt', 'w+', encoding='utf-8') \
                            as sfd:
                with open('./support/error_vins.txt', 'w+', encoding='utf-8') \
                                as efd:
                    # 进口车目录
                    folder_name = './support/es_images_0926'
                    base_path = Path(folder_name)
                    oprr_num = WxsDsm.generate_fds_test_samples(
                                oprr_num, vin_code_bmy_id_dict, 
                                bmy_id_bmy_name_dict, 
                                brand_set, base_path, sfd, efd)
        print('已经处理品牌数：{0};'.format(len(brand_set)))

    @staticmethod
    def generate_fds_test_samples(oprr_num, vin_code_bmy_id_dict, 
                bmy_id_bmy_name_dict, brand_set, base_path, sfd, efd):
        brand_num = 0
        for sub_obj in base_path.iterdir():
            filename = str(sub_obj)
            item_name = filename.split('/')[-1]
            if not sub_obj.is_dir() and filename.endswith(
                        ('jpg','png','jpeg','bmp')) and not \
                            item_name.startswith('白') \
                            and not item_name.startswith('夜'): 
                            # 忽略其下目录
                oprr_num = WxsDsm.process_one_img_file(
                            oprr_num, vin_code_bmy_id_dict, 
                            bmy_id_bmy_name_dict, brand_set, 
                            sub_obj, sfd, efd)
        return oprr_num

    @staticmethod
    def bind_fds_brand_head_bmy_head():
        '''
        实现先预测出品牌类别，然后从年款头中除该品牌对应的年款索引外的其他
        类别全部清零，将年款头的内容输出作为输出
        '''
        # 列出年款文件./support/cambricon_vehicle_label.txt内容
        brand_idx_bmys = {}
        bmys = CBmy.get_bmys()
        for bmy in bmys:
            bmy_id = int(bmy['bmy_id'])
            brand_id = int(bmy['brand_id'])
            if brand_id in brand_idx_bmys:
                brand_idx_bmys[brand_id].append(bmy_id)
            else:
                brand_idx_bmys[brand_id] = [bmy_id]
        '''
        id_bmy_dict = {}
        row = 0
        with open('./support/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                line = line.strip()
                id_bmy_dict[row] = line
                row += 1
        # 获取品牌列表
        brand_idx_bmys = {}
        with open('./support/bid_brands_dict.txt', 'r', encoding='utf-8') as bfd:
            for line in bfd:
                line = line.strip()
                arrs0 = line.split(':')
                brand_idx = int(arrs0[0])
                brand_name = arrs0[1]
                brand_idx_bmys[brand_idx] = []
                for k, v in id_bmy_dict.items():
                    if v.startswith(brand_name):
                        brand_idx_bmys[brand_idx].append(k)
        '''
        # 生成Python代码LoadModel.py中的结构代码
        print('{')
        for k, v in brand_idx_bmys.items():
            print('{0}:{1},'.format(k, v))
        print('}')
        # 生成C++后处理中的Vector初始化代码
        print('############################')
        print('{')
        for k, v in brand_idx_bmys.items():
            row_str = '{'
            first_item = True
            for vi in v:
                if first_item:
                    row_str += '{0}'.format(vi)
                    first_item = False
                else:
                    row_str += ',{0}'.format(vi)
            row_str += '},'
            print(row_str)
        print('}')
        
    @staticmethod
    def compare_new_wxs_bid_excel():
        '''
        检查所里9月18日发布的文件中，仅有2200个品牌车型的代码，与之前版本的
        品牌车型年款代码进行比较，看看是否一致
        '''
        print('检查新版所里Excel文件是否有变化...')
        wxs_db_bm_dict = CModel.get_wxs_db_bm_dict()
        wxs_bid_bm_dict = {}
        first_row = True
        with open('./support/wxs_bid_brand_model.csv', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                if first_row:
                    first_row = False
                    continue
                line = line.strip()
                arrs_a = line.split(',')
                bm_code = arrs_a[2]
                bm_name = '{0}-{1}'.format(arrs_a[1].strip(), arrs_a[3].strip())
                wxs_bid_bm_dict[bm_code] = bm_name
        for k, v in wxs_bid_bm_dict.items():
            dbv = wxs_db_bm_dict[k]
            if v != dbv:
                print('{0}: 标书：{1}；数据库：{2};'.format(k, v, dbv))
                
    @staticmethod
    def vd_cut_by_folder():
        '''
        对指定目录文件进行切图，以树形目录方式存到目录中，将切图失败的图片
        保存到另外的目录
        '''
        VdJsonManager.vd_cut_by_folder()
        
    @staticmethod
    def refine_fds_test_ds():
        '''
        将全量数据集测试集图片换为切过的图片
        '''
        cutted_img_dict = {}
        for root, dirs, files in os.walk('./support/esi0926_cutted', topdown=False):
            for fn in files:
                full_fn = '{0}/{1}'.format(root, fn)
                cutted_img_dict[fn] = full_fn
        error_num = 0
        with open('./support/fds_test_ds.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/fds_test_ds_old.txt', 'r', encoding='utf-8') as ofd:
                for line in ofd:
                    line = line.strip()
                    arrs_a = line.split('*')
                    org_full_fn = arrs_a[0]
                    bmy_id = arrs_a[1]
                    brand_id = arrs_a[2]
                    arrs_b = org_full_fn.split('/')
                    img_file = arrs_b[-1]
                    if img_file in cutted_img_dict:
                        full_fn = cutted_img_dict[img_file]
                        print('{0}*{1}*{2};'.format(full_fn, bmy_id, brand_id))
                        wfd.write('{0}*{1}*{2}\n'.format(full_fn, bmy_id, brand_id))
                    else:
                        print('切图失败文件：{0};'.format(org_full_fn))
                        error_num += 1
        print('切图失败文件数：{0};'.format(error_num))
        
        
    @staticmethod
    def wxs_bid_ds_main():
        print('无锡所招标测试集数据预处理程序...')
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        sfd = open('./support/wxs_train_ds.txt', 'w+', encoding='utf-8')
        efd = open('./support/wxs_ds_error.txt', 'w+', encoding='utf-8')
        num = 0
        for root, dirs, files in os.walk('./support/ds_files/wxs_ds', topdown=False):
            for fn in files:
                if fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                    full_fn = '{0}/{1}'.format(root, fn)
                    WxsDsm.process_one_wxs_bid_ds_image(vin_code_bmy_id_dict, bmy_id_bmy_vo_dict, full_fn, sfd, efd)
                    num += 1
        print('共有{0}个文件'.format(num))
        sfd.close()
        efd.close()

    @staticmethod
    def process_one_wxs_bid_ds_image(vin_code_bmy_id_dict, bmy_id_bmy_vo_dict, full_fn, sfd, efd):
        '''
        对每个图片进行处理（不包括所里原来的品牌测试集图片）
        '''
        arrs_a = full_fn.split('/')
        fn = arrs_a[-1]
        arrs_b = fn.split('_')
        arrs_c = arrs_b[0].split('#')
        vin_code = arrs_c[0]
        if vin_code in vin_code_bmy_id_dict:
            bmy_id = vin_code_bmy_id_dict[vin_code]
        else:
            vin_had_bmy_id = False
            for k, _ in vin_code_bmy_id_dict.items():
                if k.startswith(vin_code):
                    bmy_id = vin_code_bmy_id_dict[k]
                    vin_had_bmy_id = True
                    break
            if not vin_had_bmy_id:
                bmy_id = -1
                if vin_code != '白' and vin_code != '夜':
                    efd.write('{0}\n'.format(full_fn))
        if bmy_id > 0:
            bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
            sfd.write('{0}*{1}*{2}\n'.format(full_fn, bmy_id, bmy_vo['brand_id']))
            
    @staticmethod
    def form_wxs_bid_test_rst():
        print('生成无锡所数据集测试结果...v0.0.1')
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        efd = open('./support/wxs_bid_test_rst_error.txt', 'w+', encoding='utf-8')
        with open('./support/wxs_ds_rst.txt', 'w+', encoding='utf-8') as wfd:
            with open('./support/wxs_ds_images.txt', 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs_a = line.split('/')
                    img_file = arrs_a[-1]
                    arrs_b = img_file.split('_')
                    arrs_c = arrs_b[0].split('#')
                    vin_code = arrs_c[0]
                    if vin_code in vin_code_bmy_id_dict:
                        bmy_id = vin_code_bmy_id_dict[vin_code]
                    else:
                        vin_had_bmy_id = False
                        for k, _ in vin_code_bmy_id_dict.items():
                            if k.startswith(vin_code):
                                bmy_id = vin_code_bmy_id_dict[k]
                                vin_had_bmy_id = True
                                break
                        if not vin_had_bmy_id:
                            bmy_id = -1
                            if vin_code != '白' and vin_code != '夜':
                                efd.write('{0}\n'.format(img_file))
                    if bmy_id > 0:
                        bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                        wfd.write('{0}*{1}*{2}\n'.format(img_file, bmy_vo['brand_code'], bmy_vo['model_code']))
        efd.close()
        
    @staticmethod
    def get_wxs_bid_ds_scores():
        print('计算在无锡所数据集上的品牌精度和车型精度')
        brand_acc_dict = {}
        bm_acc_dict = {}
        with open('./support/wxs_ds_rst.txt', 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[0]
                brand_code = arrs_a[1]
                bm_code = arrs_a[2]
                brand_acc_dict[img_file] = brand_code
                bm_acc_dict[img_file] = bm_code
        base_path = Path('./support/wxs_ds_rst')
        brand_num = 0
        bm_num = 0
        total_num = 0
        for fo in base_path.iterdir():
            full_fn = str(fo)
            arrs_a = full_fn.split('/')
            json_file = arrs_a[-1]
            arrs_b = json_file.split('_')
            img_file = '{0}_{1}_{2}_{3}_{4}'.format(
                arrs_b[0], arrs_b[1], arrs_b[2], arrs_b[3], arrs_b[4]
            )
            print('full_fn: {0}'.format(full_fn))
            with open(full_fn, 'r', encoding='utf-8') as jfd:
                vd_json = jfd.read()
            data = json.loads(vd_json) # VdJsonManager.get_img_reid_feature_vector(full_fn)
            clpp, ppcx, ppxhms = WxsDsm.parse_vd_json_for_ppcx(data)
            brand_code = brand_acc_dict[img_file]
            bm_code = bm_acc_dict[img_file]
            total_num += 1
            if brand_code == clpp:
                brand_num += 1
            if bm_code == ppcx:
                bm_num += 1
        print('brand_acc: {0};'.format(brand_num / total_num))
        print('bm_acc: {0};'.format(bm_num / total_num))
            
    @staticmethod
    def parse_vd_json_for_ppcx(data):
        cllxfls = ['11', '12', '13', '14', '21', '22']
        if ('VEH' not in data) or len(data['VEH']) < 1:
            return None, None, None
        else:
            # 找到面积最大的检测框作为最终检测结果
            max_idx = -1
            max_area = 0
            for idx, veh in enumerate(data['VEH']):
                cllxfl = veh['CXTZ']['CLLXFL'][:2]
                if cllxfl in cllxfls:
                    box_str = veh['WZTZ']['CLWZ']
                    arrs_a = box_str.split(',')
                    x1, y1, w, h = int(arrs_a[0]), int(arrs_a[1]), int(arrs_a[2]), int(arrs_a[3])
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_idx = idx
            if max_idx < 0:
                return None, None, None
            else:
                return data['VEH'][max_idx]['CXTZ']['CLPP'], data['VEH'][max_idx]['CXTZ']['PPCX'], \
                        data['VEH'][max_idx]['CXTZ']['PPXHMS']
        
    @staticmethod
    def generate_samples_wxs_bid_ds():
        vin_code_bmy_id_dict = CBmy.get_wxs_vin_code_bmy_id_dict()
        bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        bmy_id_bmy_vo = CBmy.get_bmy_id_bmy_vo_dict()
        img_base_folder = './support/ds_files/wxs_ds'
        base_path = Path(img_base_folder)
        error_num = 0
        b_num = 0
        total = 0
        with open('./support/raw_wxs_bid_train_ds.txt', 'w+', encoding='utf-8') as wfd:
            for root, dirs, files in os.walk('./support/ds_files/wxs_ds', topdown=False):
                for img_file in files:
                    full_fn = '{0}/{1}'.format(root, img_file)
                    arrs_a = img_file.split('_')
                    arrs_b = arrs_a[0].split('#')
                    vin_code = arrs_b[0]
                    #print('{0}: {1};'.format(img_file, vin_code))
                    if vin_code in vin_code_bmy_id_dict:
                        bmy_id = int(vin_code_bmy_id_dict[vin_code])
                        org_bmy_id = bmy_id - 1
                        bmy_vo = bmy_id_bmy_vo[bmy_id]
                        wfd.write('{0}*{1}\n'.format('{0}/{1}'.format(img_base_folder, img_file), org_bmy_id))
                        total += 1
                        if bmy_vo['bmy_code'].startswith('b'):
                            print('{0}: {1}-{2};'.format(img_file, bmy_id, bmy_vo['bmy_code']))
                            b_num += 1
                    else:
                        bmy_id = -1
                        error_num += 1
                        print('### Error: {0} = {1};'.format(img_file, vin_code))
        print('total: {0}; brand_error: {1}; error num: {2};'.format(total, b_num, error_num))

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
