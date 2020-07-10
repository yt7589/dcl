# 数据集管理类，负责生成数据描述文件
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

class WxsDsm(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsDsm'

    @staticmethod
    def know_init_status():
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
        folder_brand_set = set()
        db_brand_set = set()
        for brand_obj in path_obj.iterdir():
            arrs0 = str(brand_obj).split('/')
            brand_name = '{0}牌'.format(arrs0[-1])
            folder_brand_set.add(brand_name)
        bid_had_we_no = WxsDsm.g_brand_set - folder_brand_set
        print('标书书我们没有品牌：共{0}个，分别为：{1};'.format(len(bid_had_we_no), bid_had_we_no))
        print('******************************************************')
        we_had_bid_no = folder_brand_set - WxsDsm.g_brand_set
        wb_brand_dict = {}
        is_break = False
        for brand_name in we_had_bid_no:
            base_path = Path('/media/zjkj/work/fgvc_dataset/raw/{0}'.format(brand_name[:-1]))
            is_break = False
            for model_obj in base_path.iterdir():
                for year_obj in model_obj.iterdir():
                    for item_obj in year_obj.iterdir():
                        item_name = str(item_obj)
                        if not sub_obj.is_dir() and filename.endswith(
                                    ('jpg','png','jpeg','bmp')): # 忽略其下目录
                            arrs0 = item_name.split('/')
                            arrs1 = arrs0[-1].split('#')
                            vin_code = arrs1[0]
                            if vin_code in vin_bmy_id_dict:
                                bmy_id = vin_bmy_id_dict[vin_code]
                            elif vin_code[:8] in vin_bmy_id_dict:
                                bmy_id = vin_bmy_id_dict[vin_code[:8]]
                            else:
                                #wfd.write('############## {0}\n'.format(vin_code))
                                bmy_id = -1
                            print('正在处理：{0};  {1}'.format(item_name, bmy_id))
                            if bmy_id > 0:
                                bmy_name = WxsDsm.g_bmy_id_bmy_name_dict[bmy_id]
                                arrsn = bmy_name.split('-')
                                brand_name1 = arrsn[0]
                                if brand_name not in wb_brand_dict:
                                    wb_brand_dict[brand_name] = brand_name1
                                is_break = True
                                break
                    if is_break:
                        break
                if is_break:
                    break
        for k, v in wb_brand_dict.items():
            print('### {0}: {1};'.format(k, v))

        print('我们有标书没有品牌：共{0}个，分别为：{1};'.format(len(we_had_bid_no), we_had_bid_no))
        sys.exit(0)

    @staticmethod
    def generate_dataset():
        print('生成数据集...')
        vins = CBmy.get_vin_codes()
        vin_samples_dict = WxsDsm.get_vin_samples_dict()
        with open('./logs/bid_train_ds.txt', 'w+', encoding='utf-8') as train_fd:
            with open('./logs/bid_test_ds.txt', 'w+', encoding='utf-8') as test_fd:
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
                        vin_samples_dict[vin_code] = [{'img_file': arrs[0], 'bmy_id': arrs[1]}]
                    else:
                        vin_samples_dict[vin_code].append({'img_file': arrs[0], 'bmy_id': arrs[1]})
        return vin_samples_dict

    @staticmethod
    def process_bt_1000_samples(samples, train_fd, test_fd):
        '''
        随机则取10张作为测试数据集，其余作为训练数据集
        '''
        data = list(range(len(samples)))
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@1 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
        train_idxs = data[10:1011]
        print('训练数据集：')
        for idx in train_idxs:
            print('#1 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))

    @staticmethod
    def process_100_to_1000_samples(samples, train_fd, test_fd):
        '''
        随机则取10张作为测试数据集，其余作为训练数据集
        '''
        data = list(range(len(samples)))
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@2 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
        train_idxs = data[10:]
        print('训练数据集：')
        for idx in train_idxs:
            print('#2 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))

    @staticmethod
    def process_10_to_100_samples(samples, train_fd, test_fd):
        '''
        随机取10张作为测试数据集，取全部图片作为训练数据集
        '''
        data = list(range(len(samples)))
        test_idxs = data[:10]
        print('测试数据集：')
        for idx in test_idxs:
            print('@3 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
        train_idxs = data
        print('训练数据集：')
        for idx in train_idxs:
            print('#3 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))

    @staticmethod
    def process_lt_10_samples(samples, train_fd, test_fd):
        test_idxs = list(range(len(samples)))
        print('测试数据集：')
        for idx in test_idxs:
            print('@4 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 3)
            test_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
        train_idxs = test_idxs
        print('训练数据集：')
        for idx in train_idxs:
            print('#4 {0}*{1};'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))
            #CDataset.add_dataset_sample(1, samples[idx]['sample_id'], 1)
            train_fd.write('{0}*{1}\n'.format(samples[idx]['img_file'], int(samples[idx]['bmy_id'])-1))

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
    def exp001():
        bmy_id = 8
        bmy_vo = CBmy.get_bmy_by_id(bmy_id)
        print('bmy_vo:{0}; {1}'.format(type(bmy_vo), bmy_vo))


    