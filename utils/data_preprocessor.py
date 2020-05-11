# 数据预处理器
# bno 品牌编号（见所里附件，1~180）
# bn 品牌名称
# bmy 品牌-车型-年款
# vc 车辆识别码，图像文件前面的编号
# vpc 车牌

class DataPreprocessor(object):
    _v_bno_bn = None # 由品牌编号查品牌名称
    _v_bn_bno = None # 由品牌名称查品牌编号

    def __init__(self):
        self.name = 'utils.DataPreprocessor'

    @staticmethod
    def startup():
        v_bno_bn = DataPreprocessor.get_v_bno_bn()
        for k, v in v_bno_bn.items():
            print('{0}: {1}'.format(k, v))
    
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