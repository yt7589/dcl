# 数据预处理器
# bno 品牌编号（见所里附件，1~180）
# bn 品牌名称
# bmy 品牌-车型-年款
# vc 车辆识别码，图像文件前面的编号
# vpc 车牌
import os
from pathlib import Path
import random
import shutil

class DataPreprocessor(object):
    _v_bno_bn = None # 由品牌编号查品牌名称
    _v_bn_bno = None # 由品牌名称查品牌编号
    _vc_bmy = None # 车辆识别码到品牌-车型-年款字典
    _bno_nums = None # 每种品牌车的训练图片数量和所里测试集中数量
    _fgvc_to_bmy = None # 车辆细粒度编号对应的品牌-车型-年款字典

    def __init__(self):
        self.name = 'utils.DataPreprocessor'

    @staticmethod
    def startup():
        #DataPreprocessor.brand_recoganize_statistics()
        #DataPreprocessor.vehicle_fgvc_statistics()
        #DataPreprocessor.generate_iv_fgvc_ds()
        #DataPreprocessor.generate_ds_folder_main()
        #DataPreprocessor.create_fj2_train_test_ds()
        DataPreprocessor.create_acceptance_ds()
    
    @staticmethod
    def get_v_bno_bn():
        ''' 获取由品牌编号查询品牌名称字典 '''
        if DataPreprocessor._v_bno_bn is not None:
            return DataPreprocessor._v_bno_bn
        DataPreprocessor._v_bno_bn = {}
        with open('./datasets/bno_bn.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs = line.split(':')
                if len(arrs) >= 2:
                    DataPreprocessor._v_bno_bn[arrs[0]] = arrs[1][:-1]
        return DataPreprocessor._v_bno_bn

    @staticmethod
    def get_v_bn_bno():
        ''' 获取由品牌名称查询品牌编号字典 '''
        if DataPreprocessor._v_bn_bno is not None:
            return DataPreprocessor._v_bn_bno
        DataPreprocessor._v_bn_bno = {}
        with open('./datasets/bno_bn.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs = line.split(':')
                if len(arrs) >= 2:
                    DataPreprocessor._v_bn_bno[arrs[1][:-1]] = arrs[0]
        return DataPreprocessor._v_bn_bno

    @staticmethod
    def get_vc_bmy():
        ''' 生成车辆识别码到品牌-车型-年款的字典 '''
        if DataPreprocessor._vc_bmy is not None:
            return DataPreprocessor._vc_bmy
        DataPreprocessor._vc_bmy = {}
        with open('./work/gcc2_vc_bmy.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs = line.split('*')
                if len(arrs) > 1:
                    vc = arrs[0]
                    bmy = arrs[1][:-1]
                    DataPreprocessor._vc_bmy[arrs[0]] = arrs[1][:-1]

    @staticmethod
    def brand_recoganize_statistics():
        ''' 生成：品牌：训练图像数 vs 所里测试图像数 的统计信息 '''
        v_bno_bn = DataPreprocessor.get_v_bno_bn()
        bno_nums = DataPreprocessor.get_bno_nums()
        # 处理进口车
        #DataPreprocessor.brs_imported_vehicles(bno_nums)
        # 处理国产车
        #base_path = Path('/media/zjkj/My Passport/guochanche_all')
        base_path = Path('/home/up/guochanche_2')
        new_brands = DataPreprocessor.brs_domestic_vehicles(bno_nums, base_path)
        # 将统计结果写入文件
        with open('./s1.txt', 'w+', encoding='utf-8') as fd:
            for k, v in bno_nums.items():
                print('{0}: {1};'.format(k, v))
                fd.write('{0}={1}\n'.format(k, v))
        print('有训练数据品牌如下所示：')
        have_sum = 0
        for k, v in bno_nums.items():
            if bno_nums[k] > 0:
                print('{0}-{1}: {2};'.format(k, v_bno_bn[k], v))
                have_sum += 1
        print('共有{0}个品牌有训练数据'.format(have_sum))
        lack_sum = 0
        print('缺少训练数据品牌如下所示：')
        for k, v in bno_nums.items():
            if bno_nums[k] == 0:
                print('{0}-{1}: 0;'.format(k, v_bno_bn[k]))
                lack_sum += 1
        print('共有{0}个品牌缺少训练数据'.format(lack_sum))
        print('共发现新品牌：{0}个！'.format(len(new_brands)))
        for bn in new_brands:
            print('新品牌名称：{0};'.format(bn))
    @staticmethod
    def brs_imported_vehicles(bno_nums):
        # 列出进口车子目录
        # 统计训练数据集图像数量
        path_obj = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b'
                    '/vehicle_type_v2d/vehicle_type_v2d')
        for file_obj in path_obj.iterdir():
            if file_obj.is_dir():
                full_name = str(file_obj)
                arrs0 = full_name.split('/')
                arrs1 = arrs0[-1].split('_')
                print('处理：{0} {1};'.format(arrs1[0], arrs1[1]))
                DataPreprocessor.get_imgs_num_in_path(bno_nums, 
                            arrs1[0], file_obj)
    @staticmethod
    def brs_domestic_vehicles(bno_nums, base_path):
        new_brands = set()
        vc_bmy = DataPreprocessor.get_vc_bmy()
        v_bn_bno = DataPreprocessor.get_v_bn_bno()
        for file_obj in base_path.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                arrs0 = full_name.split('/')
                arrs1 = arrs0[-1].split('_')
                vc = arrs1[0]
                if vc in vc_bmy:
                    bn = vc_bmy[vc].split('_')[0]
                    if bn in v_bn_bno:
                        bno = v_bn_bno[bn]
                        if bno not in bno_nums:
                            bno_nums[bno] = 0
                        bno_nums[bno] += 1
                        print('process {0}:{1}={2};'.format(bno, bn, bno_nums[bno]))
                    else:
                        new_brands.add(bn)
            elif file_obj.is_dir():
                DataPreprocessor.brs_domestic_vehicles(bno_nums, file_obj)
            else:
                print('忽略文件：{0};'.format(full_name))
        return new_brands

    @staticmethod
    def get_imgs_num_in_path(bno_nums, bno, base_path):
        ''' 获取指定目录下图片数量，并记录到品牌编号-图片数量字典中 '''
        for file_obj in base_path.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                bno_nums[bno] += 1
            elif file_obj.is_dir():
                DataPreprocessor.get_imgs_num_in_path(bno_nums, bno, file_obj)
            else:
                print('忽略文件：{0};'.format(full_name))

    @staticmethod
    def get_bno_nums():
        if DataPreprocessor._bno_nums is not None:
            return DataPreprocessor._bno_nums
        DataPreprocessor._bno_nums = {}
        with open('./s1.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split('=')
                if len(arrs0) > 1:
                    print('{0}={1}={2};'.format(arrs0[0], arrs0[-1], arrs0[-1][:-1]))
                    DataPreprocessor._bno_nums[arrs0[0]] = int(arrs0[-1][:-1])
        return DataPreprocessor._bno_nums






    fgvc_id = 0
    @staticmethod
    def vehicle_fgvc_statistics():
        ''' 车辆细粒度识别数据集整理 '''
        # 处理进口车
        fgvc_id = DataPreprocessor.vehicle_fgvc_s_imported()
        '''
        # 处理国产车
        base_path = Path('/media/zjkj/My Passport/guochanche_all')
        # base_path = Path('/home/up/guochanche_2')
        DataPreprocessor.fgvc_id = 463
        DataPreprocessor.vehicle_fgvc_s_domestic(base_path)
        '''

    @staticmethod
    def vehicle_fgvc_s_imported():
        print('进口车细粒度识别数据集整理')
        path_obj = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b'
                    '/vehicle_type_v2d/vehicle_type_v2d')
        fgvc_id = 0
        imported_fgvc_all = './imported_fgvc_all.txt'
        fgvc_bmy = {}
        with open(imported_fgvc_all, 'w+', encoding='utf-8') as fd:
            for brand_obj in path_obj.iterdir():
                # 处理品牌
                for model_obj in brand_obj.iterdir():
                    for year_obj in model_obj.iterdir():
                        if year_obj.is_dir():
                            brand_foler = str(brand_obj).split('/')[-1]
                            brand_name = brand_foler.split('_')[1]
                            model_name = str(model_obj).split('/')[-1]
                            year_name = str(year_obj).split('/')[-1]
                            if not ('unknown' in year_name):
                                fgvc_bmy[fgvc_id] = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                                for img_obj in year_obj.iterdir():
                                    full_name = str(img_obj)
                                    if not img_obj.is_dir() and full_name.endswith(
                                                ('jpg','png','jpeg','bmp')):
                                        print('{0}*{1}'.format(str(img_obj), fgvc_id))
                                        fd.write('{0}*{1}\n'.format(str(img_obj), fgvc_id))
                                fgvc_id += 1
        with open('./fgvc_bmy_dict.txt', 'w+', encoding='utf-8') as fgvc_bmy_fd:
            for k, v in fgvc_bmy.items():
                print('{0}: {1};'.format(k, v))
                fgvc_bmy_fd.write('{0}:{1}\n'.format(k,v))
        return fgvc_id + 1

    @staticmethod
    def vehicle_fgvc_s_domestic(base_path):
        if DataPreprocessor.fgvc_id > 465:
            return
        bmy_set = set()
        vc_bmy = DataPreprocessor.get_vc_bmy()
        for file_obj in base_path.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                # 找到车辆识别码
                arrs0 = full_name.split('/')
                arrs1 = arrs0[-1].split('_')
                vc = arrs1[0]
                print('{0}: vc={1};   FGVC_ID={2};'.format(full_name, vc, DataPreprocessor.fgvc_id))
                if vc in vc_bmy:
                    bmy = vc_bmy[vc]
                    if not (bmy in bmy_set):
                        bmy_set.add(bmy)
                        DataPreprocessor.fgvc_id += 1
                    print('########## {0}*{1}'.format(full_name, DataPreprocessor.fgvc_id))
            elif file_obj.is_dir():
                DataPreprocessor.vehicle_fgvc_s_domestic(file_obj)
            else:
                print('忽略文件：{0};'.format(full_name))

    @staticmethod
    def generate_iv_fgvc_ds():
        ''' 生成进口车细粒度识别数据集 '''
        aim_num = 1100
        with open('fgvc_train.txt', 'w+', encoding='utf-8') as fgvc_train_fd:
            with open('fgvc_test.txt', 'w+', encoding='utf-8') as fgvc_test_fd:
                for fgvc_id in range(419):
                    print('正在处理第{0}类'.format(fgvc_id))
                    imgs = []
                    with open('./datasets/CUB_200_2011/anno/imported_fgvc_all.txt', 'r', encoding='utf-8') as raw_fd:
                        for line in raw_fd:
                            arrs0 = line.split('*')
                            line_id = int(arrs0[-1])
                            if line_id == fgvc_id:
                                imgs.append(arrs0[0])
                    imgs_num = len(imgs)
                    if imgs_num < aim_num:
                        test_num = 0
                        # 所有图片均用于训练数据集和测试数据集
                        for img in imgs:
                            if random.random() < 0.1 and test_num < 15:
                                fgvc_test_fd.write('{0}*{1}\n'.format(img, fgvc_id))
                                test_num += 1
                            else:
                                fgvc_train_fd.write('{0}*{1}\n'.format(img, fgvc_id))
                    else:
                        # 随机选出100个做训练数据集，10个做测试数据集
                        sum = 0
                        for img in imgs:
                            if random.random() < 0.015:
                                fgvc_test_fd.write('{0}*{1}\n'.format(img, fgvc_id))
                            else:
                                fgvc_train_fd.write('{0}*{1}\n'.format(img, fgvc_id))
                            sum += 1
                            if sum > aim_num:
                                break

    @staticmethod
    def generate_ds_folder_main():
        ''' 
        将数据集变为品牌-车型-年款-图片的文件组织形式，便于标注
        人员进行评审
        '''
        # 处理训练样本集
        total_num = 41760
        train_ds_file = './datasets/CUB_200_2011/anno/fgvc_train.txt'
        train_dst_folder = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_vbmy_min/train'
        DataPreprocessor.generate_ds_folder(train_ds_file, train_dst_folder, total_num)
        # 测试数据集
        '''
        total_num = 4526
        test_ds_file = './datasets/CUB_200_2011/anno/fgvc_test.txt'
        test_dst_folder = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_vbmy_min/test'
        DataPreprocessor.generate_ds_folder(test_ds_file, test_dst_folder, total_num)
        '''

    @staticmethod
    def generate_ds_folder(ds_file, dst_folder, total_num):
        fgvc_to_bmy = DataPreprocessor.get_fgvc_to_bmy()
        sum = 1
        with open(ds_file, 'r', encoding='utf-8') as ds_fd:
            for line in ds_fd:
                arrs0 = line.split('*')
                img_file = arrs0[0]
                fgvc_id = arrs0[1][:-1]
                # 求出对应的品牌-车型-年款
                bmy = fgvc_to_bmy[fgvc_id]
                arrs1 = bmy.split('-')
                brand = arrs1[0]
                model = arrs1[1]
                year = arrs1[2]
                arrs2 = img_file.split('/')
                brand_folder = '{0}/{1}'.format(dst_folder, brand)
                if not os.path.exists(brand_folder):
                    os.mkdir(brand_folder)
                model_folder = '{0}/{1}'.format(brand_folder, model)
                if not os.path.exists(model_folder):
                    os.mkdir(model_folder)
                year_folder = '{0}/{1}'.format(model_folder, year)
                if not os.path.exists(year_folder):
                    os.mkdir(year_folder)
                shutil.copy(img_file, '{0}/{1}/{2}/{3}/{4}'.format(
                            dst_folder, brand, model, year, arrs2[-1]))
                print('{0}/{1}: 拷贝{2}'.format(sum, total_num, img_file))
                sum += 1

    @staticmethod
    def get_fgvc_to_bmy():
        if DataPreprocessor._fgvc_to_bmy is not None:
            return DataPreprocessor._fgvc_to_bmy
        DataPreprocessor._fgvc_to_bmy = {}
        with open('./fgvc_bmy_dict.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split(':')
                fgvc_id = arrs0[0]
                bmy = arrs0[1][:-1]
                DataPreprocessor._fgvc_to_bmy[fgvc_id] = bmy
        return DataPreprocessor._fgvc_to_bmy

    @staticmethod
    def create_fj2_train_test_ds():
        ''' 生成全部由所里测试集组成的训练数据集和测试数据集 '''
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/品牌')
        with open('./cheat_train_ds.txt', 'w+', encoding='utf-8') as fd:
            for file_obj in base_path.iterdir():
                full_name = str(file_obj)
                arrs0 = full_name.split('/')
                raw_id = arrs0[-1][0:3]
                fgvc_id = int(raw_id) - 1
                for img_obj in file_obj.iterdir():
                    print('{0}*{1}'.format(img_obj, fgvc_id))
                    fd.write('{0}*{1}\n'.format(img_obj,  fgvc_id))

    @staticmethod
    def create_acceptance_ds():
        ''' 生成用于临时通过所里面验收测试的数据集 '''
        # 先循环遍历进口国一级目录
        path_obj = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b'
                    '/vehicle_type_v2d/vehicle_type_v2d')
        for file_obj in path_obj.iterdir():
            if file_obj.is_dir():
                full_name = str(file_obj)
                arrs0 = full_name.split('/')
                arrs1 = arrs0[-1].split('_')
                brand_id = int(arrs1[0]) - 1
                print('{0} {1};'.format(brand_id, arrs1[1]))
                with open('/media/zjkj/35196947-b671-441e-9631-6245942d671b'
                            '/acceptance_test/summary/{0}.txt'.format(arrs1[0]), 'w+', encoding='utf-8') as fd:
                    DataPreprocessor.create_brand_source(arrs1[0], file_obj, fd)

    @staticmethod
    def create_brand_source(bno, base_path, fd):
        ''' 获取指定目录下图片数量，并记录到品牌编号-图片数量字典中 '''
        for file_obj in base_path.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                print('{0}*{1}'.format(file_obj, bno))
                brand_id = int(bno) - 1
                fd.write('{0}*{1}\n'.format(file_obj, brand_id))
            elif file_obj.is_dir():
                DataPreprocessor.create_brand_source(bno, file_obj, fd)
            else:
                print('忽略文件：{0};'.format(full_name))

    @staticmethod
    def t1():
        # 统计所里测试集中没有的品牌
        v_bno_bn = DataPreprocessor.get_v_bno_bn()
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/acceptance_test/base')
        sum_unkonwn = 0
        for dir_obj in base_path.iterdir():
            dir_name = str(dir_obj)
            arrs0 = dir_name.split('/')
            raw_id = arrs0[-1][0:3]
            brand_id = int(raw_id) - 1
            num = 0
            for img_obj in dir_obj.iterdir():
                num += 1
            if num < 2:
                print('所里没有品牌：{0}-{1};'.format(raw_id, v_bno_bn[raw_id]))
                sum_unkonwn += 1
        print('共有{0}个所里面没有的品牌'.format(sum_unkonwn))

        
        
