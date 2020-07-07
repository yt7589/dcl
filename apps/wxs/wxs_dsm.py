# 数据集管理类，负责生成数据描述文件
import sys
import shutil
from pathlib import Path
from apps.wxs.controller.c_brand import CBrand
from apps.wxs.controller.c_model import CModel
from apps.wxs.controller.c_bmy import CBmy
from apps.wxs.controller.c_sample import CSample

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

    @staticmethod
    def generate_samples():
        folder_name = '/media/zjkj/work/fgvc_dataset/raw'
        base_path = Path(folder_name)
        WxsDsm.generate_samples_from_path(base_path)

    @staticmethod
    def generate_samples_from_path(path_obj):
        for sub_obj in path_obj.iterdir():
            sub_file = str(sub_obj)
            if sub_obj.is_dir():
                WxsDsm.generate_samples_from_path(sub_obj)
            elif sub_file.endswith(('jpg','png','jpeg','bmp')):
                print('处理文件：{0};'.format(sub_obj))

    @staticmethod
    def exp001():
        vin_id = 1
        bmy_id = 2
        img_file = 'a.jpg'
        CSample.add_sample(img_file, vin_id, bmy_id)

    