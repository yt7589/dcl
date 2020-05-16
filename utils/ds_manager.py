# 数据集管理器
# 1. 进口车数据集处理方法
# 原始数据集目录，从原始数据集中每个细分类取出100张训练图片和10张测试图片，组成精简数据集，剩余部分为残差数据集。将精简数据集进行人工核对后，在精简数据集上运行DCL模型达到90~95%的精度。
# 然后再从列差数据集中每个细分类中挑出100个训练图片和10个测试图片，经过人工检验后，加入到精简数据集中，重新用DCL进行训练，达到90~95%精度。
# 重复以上过程，直到将所有进口车数据全部处理完成为止。
# 2. 国产车数据处理方法
# 首先将图片开头编号中没有对应到品牌-车型-年款的记录通过网络查询，找到对应关系。
# 按照对应关系，将国产车数据整理为品牌/车型/年款的目录格式，采用与进口车同样的方式找出精简数据集，逐步迭代出完整的数据集。
# 术语：
# bmy: 品牌-车型-年款
from pathlib import Path

class DsManager(object):
    _fgvc_id_bmy_dict = None # 细分类编号到品牌-车型-年款字典
    _bmy_fgvc_id_dict = None # 品牌-车型-年款到细分类编号字典

    def __init__(self):
        self.name = 'utils.DsManager'

    @staticmethod
    def startup():
        DsManager.sample_imported_vehicle_data()

    @staticmethod
    def sample_imported_vehicle_data():
        '''
        遍历yantao/rest_data/目录，第一级为品牌，第二层为车型，第三层为年款，利用三重循环
        1. 形成“品牌-车型-年款”字符串，从_bmy_fgvc_id_dict中查出fgvc_id编号，如果没有则创建；
        2. 将其下文件形成一个列表，并求出总数；
        3. 循环处理这个列表：
        3.1. 如果总数小于100条，则全部提取为训练和测试用数据集：生成一个0~1间随机数，小于0.1时
             将其拷贝到测试集目录对应文件夹下，否则将其拷贝到训练集目录对应目录下；
        3.2. 如果总数大于100条，以100/总数的概率选择样本，然后再以10%概率作为测试集，其余作为
             训练集
        将新生成的测试集和训练集给人工进行校对
        '''
        path_obj = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b'
                    '/vehicle_type_v2d/vehicle_type_v2d')
        for dir1_obj in path_obj.iterdir():
            dir1_name = str(dir1_obj)
            arrs0 = dir1_name.split('_')
            bn = arrs0[-1]
            print('品牌：{0}; {1};'.format(bn, dir1_name))
            for dir2_obj in dir1_obj.iterdir():
                dir2_name = str(dir2_obj)
                print('model: {0};'.format(dir2_name))
    
