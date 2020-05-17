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
import random

class DsManager(object):
    _fgvc_id_bmy_dict = None # 细分类编号到品牌-车型-年款字典
    _bmy_fgvc_id_dict = None # 品牌-车型-年款到细分类编号字典
    _bmy_to_fgvc_id_dict = None
    _fgvc_id_to_bmy_dict = None

    def __init__(self):
        self.name = 'utils.DsManager'

    @staticmethod
    def startup():
        #DsManager.sample_imported_vehicle_data()
        # ************************************************
        # folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train'
        #DsManager.refine_bmy_and_fgvc_id_dicts(folder_name)
        # ***************************************************
        folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train'
        #folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test'
        ds_file = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train_ds_v1.txt'
        #ds_file = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test_ds_v1.txt'
        DsManager.generate_ds_by_folder(folder_name, ds_file)

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
        brand_num = 0
        model_num = 0
        year_num = 0
        aim_num = 100 # 品牌-车型-年款中取出的图片数
        for dir1_obj in path_obj.iterdir():
            dir1_name = str(dir1_obj)
            arrs0 = dir1_name.split('_')
            bn = arrs0[-1]
            print('品牌：{0}; {1};'.format(bn, dir1_name))
            brand_num += 1
            if not dir1_obj.is_dir() or dir1_name[-7:] == 'unknown':
                continue
            for dir2_obj in dir1_obj.iterdir():
                dir2_name = str(dir2_obj)
                print('#### model: {0};'.format(dir2_name))
                if not dir2_obj.is_dir() or dir2_name[-7:] == 'unknown':
                    continue
                model_num += 1
                for dir3_obj in dir2_obj.iterdir():
                    dir3_name = str(dir3_obj)
                    if not dir3_obj.is_dir() or dir3_name[-7:] == 'unknown':
                        continue
                    imgs_num = DsManager.get_imgs_num_in_folder(dir3_obj)
                    DsManager.sample_in_folder(dir3_obj, imgs_num, aim_num, 0.1)
                    print('******** year: {0}; imgs_num={1};'.format(dir3_name, imgs_num))
                    year_num += 1
        print('共有{0}个品牌，{1}个车型，{2}个年款'.format(brand_num, model_num, year_num))

    @staticmethod
    def sample_in_folder(folder_obj, imgs_num, aim_num, ratio):
        '''
        从指定目录中选取数据集文件，将文件从本目录移动到训练或测试集目录下的对应目录
            imgs_num：当前目录下图片文件总数
            aim_num：希望取的张数
            ratio：多大比例用于测试集
        '''
        for file_obj in folder_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                if imgs_num < aim_num:
                    # 取全部图像，其中10%作为测试集，90%作为训练集
                    if random.random() < ratio:
                        # 移动到测试集
                        print('移动到测试集')
                    else:
                        # 移动到训练集
                        print('移动到训练集')
                else:
                    # 取 aim_num / imgs_num 比例图片，其中10%作为测试集
                    if random.random() < aim_num / imgs_num:
                        if random.random() < ratio:
                            print('移动到测试集')
                        else:
                            print('移动到训练集')
            else:
                print('忽略文件：{0};'.format(file_obj))
            

    @staticmethod
    def get_imgs_num_in_folder(folder_obj):
        imgs_num = 0
        for file_obj in folder_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                imgs_num += 1
            else:
                print('忽略文件：{0};'.format(file_obj))
        return imgs_num
    

    @staticmethod
    def get_bmy_and_fgvc_id_dicts():
        '''
        从work/bym_to_fgvc_id_dict.txt和work/fgvc_id_to_bym_dict.txt文件
        中读出内容到bym_to_fgvc_id_dict和fgvc_id_to_bym_dict中。
        将指定文件夹（品牌/车型/年款），遍历到每个品牌-车型-年款组合，查询
        是否在bym_to_fgvc_id_dict和fgvc_id_to_bym_dict，如果不在则添加
        该条目，最后将内容存储到work目录对应的文件中。
        '''
        if not (DsManager._bmy_to_fgvc_id_dict is None):
            return DsManager._bmy_to_fgvc_id_dict, DsManager._fgvc_id_to_bmy_dict
        DsManager._bmy_to_fgvc_id_dict = {}
        DsManager._fgvc_id_to_bmy_dict = {}
        with open('./work/bmy_to_fgvc_id_dict.txt', 'r', encoding='utf-8') as bfi_fd:
            for line in bfi_fd:
                arrs0 = line.split(':')
                DsManager._bmy_to_fgvc_id_dict[arrs0[0]] = arrs0[1][:-1]
        with open('./work/fgvc_id_to_bmy_dict.txt', 'r', encoding='utf-8') as fib_fd:
            for line in fib_fd:
                arrs0 = line.split(':')
                DsManager._fgvc_id_to_bmy_dict[arrs0[0]] = arrs0[1][:-1]
        return DsManager._bmy_to_fgvc_id_dict, DsManager._fgvc_id_to_bmy_dict

    @staticmethod
    def refine_bmy_and_fgvc_id_dicts(folder_name):
        '''
        从work/bym_to_fgvc_id_dict.txt和work/fgvc_id_to_bmy_dict.txt文件
        中读出内容到bym_to_fgvc_id_dict和fgvc_id_to_bym_dict中。
        将指定文件夹（品牌/车型/年款），遍历到每个品牌-车型-年款组合，查询
        是否在bym_to_fgvc_id_dict和fgvc_id_to_bym_dict，如果不在则添加
        该条目，最后将内容存储到work目录对应的文件中。
        '''
        file_sep = '/'
        bmy_to_fgvc_id_dict, fgvc_id_to_bmy_dict = DsManager.get_bmy_and_fgvc_id_dicts()
        max_fgvc_id = 0
        for k,v in fgvc_id_to_bmy_dict.items():
            if int(k) > max_fgvc_id:
                max_fgvc_id = int(k)
        base_path = Path(folder_name)
        for brand_path in base_path.iterdir():
            brand_fn = str(brand_path)
            arrs1 = brand_fn.split(file_sep)
            brand_name = arrs1[-1]
            if not brand_path.is_dir() or brand_name == 'unknown':
                continue
            for model_path in brand_path.iterdir():
                model_fn = str(model_path)
                arrs2 = model_fn.split(file_sep)
                model_name = arrs2[-1]
                if not model_path.is_dir() or model_name == 'unknown':
                    continue
                for year_path in model_path.iterdir():
                    year_fn = str(year_path)
                    arrs3 = year_fn.split(file_sep)
                    year_name = arrs3[-1]
                    if not year_path.is_dir() or year_name == 'unknown':
                        continue
                    bmy = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                    if not (bmy in bmy_to_fgvc_id_dict):
                        bmy_to_fgvc_id_dict[bmy] = max_fgvc_id
                        fgvc_id_to_bmy_dict[max_fgvc_id] = bmy
                        max_fgvc_id += 1
        # 保存品牌-车型-年款到细分类编号字典
        with open('./work/bmy_to_fgvc_id_dict.txt', 'w+', encoding='utf-8') as bfi_fd:
            for k, v in bmy_to_fgvc_id_dict.items():
                bfi_fd.write('{0}:{1}\n'.format(k, v))
        # 保存细分类编号到品牌-车型-年款字典
        with open('./work/fgvc_id_to_bmy_dict.txt', 'w+', encoding='utf-8') as fib_fd:
            for k, v in fgvc_id_to_bmy_dict.items():
                fib_fd.write('{0}:{1}\n'.format(k, v))
    
    @staticmethod
    def generate_ds_by_folder(folder_name, ds_file):
        '''
        以指定文件夹内容生成数据集，指定文件夹内容格式：品牌/车型/年款 目录
        层次结构，下面是
        '''
        file_sep = '/'
        bmy_to_fgvc_id_dict, fgvc_id_to_bmy_dict = DsManager.get_bmy_and_fgvc_id_dicts()
        base_path = Path(folder_name)
        with open(ds_file, 'w+', encoding='utf-8') as ds_fd:
            for brand_path in base_path.iterdir():
                brand_fn = str(brand_path)
                arrs1 = brand_fn.split(file_sep)
                brand_name = arrs1[-1]
                if not brand_path.is_dir() or brand_name == 'unknown':
                    continue
                for model_path in brand_path.iterdir():
                    model_fn = str(model_path)
                    arrs2 = model_fn.split(file_sep)
                    model_name = arrs2[-1]
                    if not model_path.is_dir() or model_name == 'unknown':
                        continue
                    for year_path in model_path.iterdir():
                        year_fn = str(year_path)
                        arrs3 = year_fn.split(file_sep)
                        year_name = arrs3[-1]
                        if not year_path.is_dir() or year_name == 'unknown':
                            continue
                        bmy = '{0}-{1}-{2}'.format(brand_name, model_name, year_name)
                        if not (bmy in bmy_to_fgvc_id_dict):
                            max_fgvc_id += 1
                            bmy_to_fgvc_id_dict[bmy] = max_fgvc_id
                            fgvc_id_to_bmy_dict[max_fgvc_id] = bmy
                        for img_obj in year_path.iterdir():
                            img_file = str(img_obj)
                            fgvc_id = bmy_to_fgvc_id_dict[bmy]
                            print('{0}*{1}'.format(img_file, fgvc_id))
                            ds_fd.write('{0}*{1}\n'.format(img_file, fgvc_id))
