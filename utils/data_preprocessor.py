# 数据预处理器
# bno 品牌编号（见所里附件，1~180）
# bn 品牌名称
# bmy 品牌-车型-年款
# vc 车辆识别码，图像文件前面的编号
# vpc 车牌
from pathlib import Path

class DataPreprocessor(object):
    _v_bno_bn = None # 由品牌编号查品牌名称
    _v_bn_bno = None # 由品牌名称查品牌编号
    _vc_bmy = None # 车辆识别码到品牌-车型-年款字典
    _bno_nums = None # 每种品牌车的训练图片数量和所里测试集中数量
    __sum = 0

    def __init__(self):
        self.name = 'utils.DataPreprocessor'

    @staticmethod
    def startup():
        DataPreprocessor.brand_recoganize_statistics()
    
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
        DataPreprocessor._bno_nums = {}
        for idx in range(183):
            bno = '{0:03d}'.format(idx+1)
            DataPreprocessor._bno_nums[bno] = 0
        bno_nums = DataPreprocessor.get_bno_nums()
        # 处理进口车
        #DataPreprocessor.brs_imported_vehicles(bno_nums)
        # 处理国产车
        base_path = Path('/media/zjkj/My Passport/guochanche_all')
        DataPreprocessor.brs_domestic_vehicles(bno_nums, base_path)
        # 将统计结果写入文件
        with open('./s1.txt', 'w+', encoding='utf-8') as fd:
            for k, v in bno_nums.items():
                print('{0}: {1};'.format(k, v))
                fd.write('{0}={1}\n'.format(k, v))
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
        if DataPreprocessor.__sum > 10:
            return
        vc_bmy = DataPreprocessor.get_vc_bmy()
        v_bn_bno = DataPreprocessor.get_v_bn_bno()
        for file_obj in base_path.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                arrs0 = full_name.split('_')
                vc = arrs0[0]
                print('process vc={0};'.format(vc))
                if vc in vc_bmy:
                    bn = vc_bmy[vc].split('*')
                    bno = v_bn_bno[bn]
                    if bno not in bno_nums:
                        bno_nums[bno] = 0
                    print('process {0} {1};'.format(bno, bn))
                    bno_nums[bno] += 1
                    DataPreprocessor.__sum += 1
            elif file_obj.is_dir():
                DataPreprocessor.brs_domestic_vehicles(bno_nums, file_obj)
            else:
                print('忽略文件：{0};'.format(full_name))

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
        DataPreprocessor.brand_recoganize_statistics()
        return DataPreprocessor._bno_nums

