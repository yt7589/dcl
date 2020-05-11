# 数据预处理器
# bno 品牌编号（见所里附件，1~180）
# bn 品牌名称
# bmy 品牌-车型-年款
# vc 车辆识别码，图像文件前面的编号
# vpc 车牌

class DataPreprocessor(object):
    _v_bno_bn = None # 由品牌编号查品牌名称
    _v_bn_bno = None # 由品牌名称查品牌编号
    _vc_bmy = None # 车辆识别码到品牌-车型-年款字典
    _brand_nums = None # 每种品牌车的训练图片数量和所里测试集中数量

    def __init__(self):
        self.name = 'utils.DataPreprocessor'

    @staticmethod
    def startup():
        DataPreprocessor.get_vc_bmy()
    
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
