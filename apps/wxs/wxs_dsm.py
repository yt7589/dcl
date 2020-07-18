# 数据集管理类，负责生成数据描述文件
import os
from os import stat
import sys
import shutil
import random
import datetime
from pathlib import Path
from apps.wxs.controller.c_brand import CBrand
from apps.wxs.controller.c_model import CModel
from apps.wxs.controller.c_bmy import CBmy
from apps.wxs.controller.c_sample import CSample
from apps.wxs.controller.c_dataset import CDataset
import PIL.Image as Image
from PIL import ImageStat

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

    g_bmy_id_bmy_name_dict = None 
    g_vin_bmy_id_dict = None
    g_brand_set = None
    g_error_num = 0
    g_cfd = None
    @staticmethod
    def generate_samples():
        vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        WxsDsm.g_bmy_id_bmy_name_dict = CBmy.get_bmy_id_bmy_name_dict()
        WxsDsm.g_vin_bmy_id_dict = CBmy.get_vin_bmy_id_dict()
        WxsDsm.g_brand_set = set()
        with open('./logs/conflicts.txt', 'w+', encoding='utf-8') as WxsDsm.g_cfd:
            with open('./logs/samples.txt', 'w+', encoding='utf-8') as sfd:
                with open('./logs/error_vins.txt', 'w+', encoding='utf-8') as efd:
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
    def process_one_img_file(vin_bmy_id_dict, sub_obj, sfd, efd):
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
        elif vin_code[:8] in vin_bmy_id_dict:
            bmy_id = vin_bmy_id_dict[vin_code[:8]]
        else:
            #wfd.write('############## {0}\n'.format(vin_code))
            bmy_id = -1
            if vin_code != '白' and vin_code != '夜':
                efd.write('{0}\n'.format(vin_code))
                WxsDsm.g_error_num += 1
                print('##### error={0} ##### {1};'.format(WxsDsm.g_error_num, sub_file))
        if bmy_id > 0:
            sfd.write('{0}*{1}\n'.format(sub_file, bmy_id - 1))
            bmy_name = WxsDsm.g_bmy_id_bmy_name_dict[bmy_id]
            arrsn = bmy_name.split('-')
            brand_name = arrsn[0]
            WxsDsm.g_brand_set.add(brand_name)
        WxsDsm.opr_num += 1
        if WxsDsm.opr_num % 1000 == 0:
            print('处理{0}条记录...'.format(
                WxsDsm.opr_num))

    @staticmethod
    def generate_dataset():
        print('生成数据集...')
        vins = CBmy.get_vin_codes()
        vin_samples_dict = WxsDsm.get_vin_samples_dict()
        with open('./logs/raw_bid_train_ds.txt', 'w+', encoding='utf-8') as train_fd:
            with open('./logs/raw_bid_test_ds.txt', 'w+', encoding='utf-8') as test_fd:
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
        bmy_id_vin_dict = CBmy.get_bmy_id_vin_dict()
        vin_samples_dict = {}
        samples = []
        with open('./logs/samples.txt', 'r', encoding='utf-8') as sfd:
            for line in sfd:
                line.strip()
                arrs = line.split('*')
                bmy_id = int(arrs[1][:-1])
                if bmy_id in bmy_id_vin_dict:
                    vin_code = bmy_id_vin_dict[bmy_id]
                    if vin_code not in vin_samples_dict:
                        vin_samples_dict[vin_code] = [{'img_file': arrs[0], 'bmy_id': bmy_id}]
                    else:
                        vin_samples_dict[vin_code].append({'img_file': arrs[0], 'bmy_id': bmy_id})
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
        with open('./logs/raw_bid_train_ds.txt', 'r', encoding='utf-8') as fd:
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
        # 生成新的训练数据集
        WxsDsm.simplify_bid_ds(org_sim_dict, './logs/bid_train_ds.txt', './logs/raw_bid_train_ds.txt')
        # 生成新的测试数据集
        WxsDsm.simplify_bid_ds(org_sim_dict, './logs/bid_test_ds.txt', './logs/raw_bid_test_ds.txt')
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
        with open('./logs/cambricon_vehicle_label.txt', 'w+', encoding='utf-8') as fd:
            for sim_bmy_id in range(len(sim_org_dict)):
                bmy_id = sim_org_dict[sim_bmy_id]
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
        with open('./logs/cambricon_vehicle_label.txt', 'r', encoding='utf-8') as fd:
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
        idx = 0
        with open(brand_ds_file, 'w+', encoding='utf-8') as bfd:
            with open(bmy_ds_file, 'r', encoding='utf-8') as yfd:
                for line in yfd:
                    line = line.strip()
                    arrs0 = line.split('*')
                    img_file = arrs0[0]
                    bmy_id = int(arrs0[1]) + 1
                    bmy_name = bmy_id_bmy_name_dict[bmy_id]
                    arrs1= bmy_name.split('-')
                    brand_name = arrs1[0]
                    if not (brand_name in brand_set):
                        brand_set.add(brand_name)
                        brand_id_brand_name_dict[idx] = brand_name
                        brand_name_brand_id_dict[brand_name] = idx
                        idx += 1
                    brand_id = brand_name_brand_id_dict[brand_name]
                    bfd.write('{0}*{1}*{2}\n'.format(img_file, bmy_id-1, brand_id))
        if is_create_brands_dict:
            with open('./logs/bid_brands_dict.txt', 'w+', encoding='utf-8') as fd:
                for k, v in brand_id_brand_name_dict.items():
                    fd.write('{0}:{1}\n'.format(k, v))
        return len(brand_set)

    @staticmethod
    def convert_to_brand_ds_main():
        brand_num = WxsDsm.convert_to_brand_ds('./logs/bid_train_ds.txt', 
                    './logs/bid_brand_train_ds.txt', 
                    is_create_brands_dict=True)
        brand_num = WxsDsm.convert_to_brand_ds('./logs/bid_test_ds.txt', 
                    './logs/bid_brand_test_ds.txt', 
                    is_create_brands_dict=False)
        print('品牌种类：{0};'.format(brand_num))

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
        print('获取所里标书信息...')
        bid_brand_set, bid_model_set, bid_bmy_set, bid_vin_set, _, _, _, _ = WxsDsm._get_bid_info()
        print('标书要求：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(bid_vin_set), len(bid_brand_set), len(bid_bmy_set)
        ))
        curr_brand_set, curr_bmy_set = WxsDsm.get_current_info()
        delta_brand = bid_brand_set - curr_brand_set
        delta_bmy = bid_bmy_set - curr_bmy_set
        print('缺失品牌数量为{0}个，分别为：'.format(len(delta_brand)))
        for brand in delta_brand:
            print('# {0};'.format(brand))
        print('缺失年款数量为{0};'.format(len(delta_bmy)))

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
    def get_g2n_vin_codes(curr_bmy_set):
        vin_code_bmy_id_dict = CBmy.get_vin_code_bmy_id_dict()
        bmy_id_bmy_vo_dict = CBmy.get_bmy_id_bmy_vo_dict()
        base_path = Path('//media/zjkj/work/guochanche_2n')
        vin_codes = []
        num = 0
        num_new_vc = 0
        num_new_bmy = 0
        for vc in base_path.iterdir():
            print('vin_code: {0};'.format(vc))
            if vc not in vin_code_bmy_id_dict:
                vin_codes.append(vc)
                num_new_vc += 1
            else:
                bmy_id = vin_code_bmy_id_dict[vc]
                bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                bmy_name = bmy_vo['bmy_name']
                if bmy_name not in curr_bmy_set:
                    vin_codes.append(vc)
                    num_new_bmy += 1
            num += 1
            if num > 10:
                break
            if num % 100 == 0:
                print('已处理：{0}个...'.format(num))
        print('新车辆识别码{0}个，新年款{1}个;'.format(num_new_vc, num_new_bmy))
        return vin_codes
        
    @staticmethod
    def exp001():
        #WxsDsm.get_simplified_bmys()
        #WxsDsm.get_fgvc_id_brand_dict()
        #WxsDsm.get_bmy_id_img_num()
        #WxsDsm.find_bad_images('37')

        #WxsDsm.get_current_state()
        curr_brand_set, curr_bmy_set = WxsDsm.get_current_info()
        vin_codes = WxsDsm.get_g2n_vin_codes(curr_bmy_set)
        '''
        for vc in vin_codes:
            print('### {0};'.format(vc))
        '''
        print('缺失{0}个：'.format(len(vin_codes)))