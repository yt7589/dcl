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
import os
import shutil
from pathlib import Path
import random

class DsManager(object):
    _fgvc_id_bmy_dict = None # 细分类编号到品牌-车型-年款字典
    _bmy_fgvc_id_dict = None # 品牌-车型-年款到细分类编号字典
    _bmy_to_fgvc_id_dict = None
    _fgvc_id_to_bmy_dict = None
    _ggh_to_bmy_dict = None

    RUN_MODE_SAMPLE_IMPORTED = 1001 # 从进口车目录随机选取数据
    RUN_MODE_REFINE = 1002 # 根据目录内容细化品牌-车型-年款与细分类编号对应表
    RUN_MODE_FGVC_DS = 1003 # 根据fgvc_dataset/train,test目录生成数据集
    RUN_MODE_BMY_STATISTICS = 1004 # 根据gcc2_vc_bmy.txt统计国产车品牌数量和车型数量
    RUN_MODE_VEHICLE_1D = 1005 # 处理VehicleID为DCL训练集格式
    RUN_MODE_DOMESTIC_DATA = 1006 # 将国产车组织成进口车目录格式；品牌-车型-年款
    # 将raw_domestic_brands.txt、gcc2_vc_bmy.txt、guochanche_all目录、
    # guochanche_2目录的公告号与品牌-车型-年款内容合并成一个文件
    RUN_MODE_MERGE_GGH_BMY = 1007 
    RUN_MODE_MERGE_BMY_FGVC_ID = 1008 # 合并品牌-车型-年款与FGVC_ID字典
    RUN_MODE_GENERATE_FGVC_DS = 1009 # 将审核好的数据目录中内容生成训练数据集或测试数据集

    def __init__(self):
        self.name = 'utils.DsManager'

    @staticmethod
    def startup():
        run_mode = DsManager.RUN_MODE_MERGE_BMY_FGVC_ID
        # refine_bmy_and_fgvc_id_dicts
        if DsManager.RUN_MODE_SAMPLE_IMPORTED == run_mode:
            # 从进口车目录随机选取数据
            DsManager.sample_imported_vehicle_data()
        elif DsManager.RUN_MODE_REFINE == run_mode:
            # 根据目录内容细化品牌-车型-年款与细分类编号对应表
            DsManager.refine()
        elif DsManager.RUN_MODE_FGVC_DS == run_mode:
            # 根据fgvc_dataset/train,test目录生成数据集
            DsManager.fgvc_ds_main()
        elif DsManager.RUN_MODE_BMY_STATISTICS == run_mode:
            # 根据gcc2_vc_bmy.txt统计国产车品牌数量和车型数量
            DsManager.bmy_statistics()
        elif DsManager.RUN_MODE_VEHICLE_1D == run_mode:
            # 处理VehicleID为DCL训练集格式
            DsManager.process_vehicle_1d()
        elif DsManager.RUN_MODE_DOMESTIC_DATA == run_mode:
            # 将国产车组织成进口车目录格式；品牌-车型-年款
            DsManager.domestic_data_main()
        elif DsManager.RUN_MODE_MERGE_GGH_BMY == run_mode:
            # 合并多个公告号与品牌_车型_年款对应关系
            DsManager.merge_ggh_bmy()
        elif DsManager.RUN_MODE_MERGE_BMY_FGVC_ID == run_mode:
            # 合并品牌-车型-年款与FGVC_ID字典
            DsManager.merge_bmy_fgvc_id()
        elif DsManager.RUN_MODE_GENERATE_FGVC_DS == run_mode:
            # 将目录内容生成训练或测试数据集
            DsManager.generate_fgvc_ds()

    @staticmethod
    def sample_imported_vehicle_data():
        ''' 从进口车目录随机选取数据 '''
        DsManager.sample_imported_vehicle_data()

    @staticmethod
    def refine():
        ''' 根据目录内容细化品牌-车型-年款与细分类编号对应表 '''
        folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train'
        DsManager.refine_bmy_and_fgvc_id_dicts(folder_name)

    @staticmethod
    def fgvc_ds_main():
        ''' 根据fgvc_dataset/train,test目录生成数据集 '''
        #folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train'
        folder_name = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test'
        #ds_file = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train_ds_v1.txt'
        ds_file = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/test_ds_v1.txt'
        DsManager.generate_ds_by_folder(folder_name, ds_file)

    @staticmethod
    def bmy_statistics():
        bmy_set = set()
        brand_set = set()
        print('国产车品牌统计')
        DsManager.domestic_bmy_statistics(bmy_set, brand_set)
        DsManager.imported_bmy_statistics(bmy_set, brand_set)
        print('我们现有品牌库共有{0}个品牌，{1}个车型（具体到年款）'.format(len(brand_set), len(bmy_set)))
        for bn in brand_set:
            print('已有品牌: {0};'.format(bn))
        obns = DsManager.get_left_brands(brand_set)
        print('未收录品牌数：{0};'.format(len(obns)))
        for obn in obns:
            print('未知品牌: {0};'.format(obn))
        

    @staticmethod
    def domestic_bmy_statistics(bmy_set, brand_set):
        ''' 根据gcc2_vc_bmy.txt统计国产车品牌数量和车型数量 '''
        with open('./work/gcc2_vc_bmy.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split('*')
                if len(arrs0) > 1:
                    bmy = arrs0[1][:-1]
                    arrs1 = bmy.split('_')
                    brand_name = arrs1[0]
                    bmy_set.add(bmy)
                    brand_set.add(brand_name)

    @staticmethod
    def imported_bmy_statistics(bmy_set, brand_set):
        file_sep = '\\'
        base_path = Path('E:/fgvc_vbmy_min/train')
        for brand_path in base_path.iterdir():
            if not brand_path.is_dir():
                continue
            brand_name = str(brand_path).split(file_sep)[-1]
            for model_path in brand_path.iterdir():
                if not model_path.is_dir():
                    continue
                model_name = str(model_path).split(file_sep)[-1]
                for year_path in model_path.iterdir():
                    if not year_path.is_dir():
                        continue
                    year_name = str(year_path).split(file_sep)[-1]
                    bmy = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                    bmy_set.add(bmy)
                    brand_set.add(brand_name)

    @staticmethod
    def get_left_brands(brand_set):
        obns = set()
        # 官方品牌库
        with open('./datasets/bno_bn.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split(':')
                obn = arrs0[-1][:-1]
                obns.add(obn)
        for bn in brand_set:
            if bn in obns:
                obns.remove(bn)
        return obns





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

    @staticmethod
    def process_vehicle_1d():
        '''
        将开源Vehicle1D数据集转换为DCL数据集
        '''
        base_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/VehicleID/VehicleID_V1.0'
        # 从model_attr.txt中获取VID对应的模型编号
        vid_to_fgvc_id_dict = {}
        with open('{0}/attribute/model_attr.txt'.format(base_dir), 'r', encoding='utf-8') as mv_fd:
            for line in mv_fd:
                arrs = line.split(' ')
                vid = arrs[0]
                fgvc_id = arrs[1][:-1]
                vid_to_fgvc_id_dict[vid] = fgvc_id
        # 打开train_list.txt文件
        with open('{0}/vehicle1d_train.txt'.format(base_dir), 'w+', encoding='utf-8') as train_fd:
            with open('{0}/vehicle1d_test.txt'.format(base_dir), 'w+', encoding='utf-8') as test_fd:
                with open('{0}/train_test_split/train_list.txt'.format(base_dir), 'r', encoding='utf-8') as tl_fd:
                    for line in tl_fd:
                        arrs = line.split(' ')
                        img = '{0}/image/{1}.jpg'.format(base_dir, arrs[0])
                        vid = arrs[1][:-1]
                        if vid in vid_to_fgvc_id_dict:
                            print('vid: {0}={1}'.format(img, vid_to_fgvc_id_dict[vid]))
                            if random.random() < 0.045:
                                test_fd.write('{0}*{1}\n'.format(img, vid_to_fgvc_id_dict[vid]))
                            else:
                                train_fd.write('{0}*{1}\n'.format(img, vid_to_fgvc_id_dict[vid]))
                        else:
                            print('没有的VID：{0};'.format(vid))

    @staticmethod
    def domestic_data_main():
        print('处理国产车目录')
        #src_base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/guochanche_all')
        src_base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/g001')
        dst_base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        DsManager.process_base_folder(src_base_path, dst_base_path)

    @staticmethod
    def process_base_folder(src_base_path, dst_base_path):
        print('将指定目录下文件移到指定目录下')
        ggh_to_bmy_dict = DsManager.get_ggh_to_bmy_dict()
        for path_obj in src_base_path.iterdir():
            folder_name = str(path_obj).split('/')[-1]
            arrs = folder_name.split('_')
            if len(arrs) > 1:
                vggh = arrs[-1]
            else:
                vggh = folder_name
            if vggh in ggh_to_bmy_dict:
                bmy = ggh_to_bmy_dict[vggh]
                print('### {0}: {1};'.format(vggh, bmy))
                DsManager.move_img_to_data_folder(path_obj, bmy, dst_base_path)
                shutil.rmtree(path_obj)
            else:
                print('未知车辆公告号：{0};'.format(vggh))

    @staticmethod
    def move_img_to_data_folder(path_obj, bmy, dst_base_path):
        for file_obj in path_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(
                        ('jpg','png','jpeg','bmp')):
                img_file = full_name.split('/')[-1]
                dst_path = DsManager.prepare_bmy_folder(dst_base_path, bmy)
                dst_file = '{0}/{1}'.format(dst_path, img_file)
                shutil.move(full_name, dst_file)
                print('移动：{0} => {1};'.format(full_name, dst_file))
            else:
                print('忽略文件：{0};'.format(full_name))


    @staticmethod
    def get_ggh_to_bmy_dict():
        if not(DsManager._ggh_to_bmy_dict is None):
            return DsManager._ggh_to_bmy_dict
        DsManager._ggh_to_bmy_dict = {}
        with open('./work/gcc2_vc_bmy.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split('*')
                if len(arrs0) > 1:
                    arrs1 = arrs0[1][:-1].split('_')
                    bmy = '{0}_{1}_{2}'.format(arrs1[0], arrs1[1], arrs1[2][:-1])
                    DsManager._ggh_to_bmy_dict[arrs0[0]] = bmy
        return DsManager._ggh_to_bmy_dict

    @staticmethod
    def prepare_bmy_folder(dst_path, bmy):
        '''
        如果目标目录中存在品牌-车型-年款目录，则返回该目录，如果没有则创建并返回该目录
        '''
        arrs = bmy.split('_')
        brand_name = arrs[0]
        model_name = arrs[1]
        year_name = arrs[2]
        brand_path = '{0}/{1}'.format(dst_path, brand_name)
        if not os.path.exists(brand_path):
            os.mkdir(brand_path)
        model_path = '{0}/{1}'.format(brand_path, model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        year_path = '{0}/{1}'.format(model_path, year_name)
        if not os.path.exists(year_path):
            os.mkdir(year_path)
        return year_path

    @staticmethod
    def merge_ggh_bmy():
        print('合并公告号与品牌-车型-年款对应关系')
        ggh_set = set()
        unknown_ggh_set = set()
        bmy_set = set()
        ggh_to_bmy_dict = {}
        txt_file = './datasets/raw_domestic_brands.txt'
        DsManager.process_interm_txt_file(txt_file, ggh_set, unknown_ggh_set, bmy_set, ggh_to_bmy_dict)
        txt_file = './work/gcc2_vc_bmy.txt'
        DsManager.process_interm_txt_file(txt_file, ggh_set, unknown_ggh_set, bmy_set, ggh_to_bmy_dict)
        print('1:共有{0}个公告号，未知公告号为{1}个，小类有{2}个！'.format(
                    len(ggh_set), len(unknown_ggh_set), len(ggh_to_bmy_dict.keys())))
        
        rst = DsManager.process_domestic_folder('/media/zjkj/35196947-b671-441e-9631-6245942d671b/guochanche_all', ggh_set, unknown_ggh_set, bmy_set, ggh_to_bmy_dict)
        print('2:共有{0}个公告号，未知公告号为{1}个，小类有{2}个！'.format(
                    len(ggh_set), len(unknown_ggh_set), len(ggh_to_bmy_dict.keys())))
        with open('./work/ggh_to_bmy_dict.txt', 'w+', encoding='utf-8') as gbd_fd:
            for k, v in ggh_to_bmy_dict.items():
                gbd_fd.write('{0}:{1}\n'.format(k, v))
        with open('./work/unknown_ggh.txt', 'w+', encoding='utf-8') as ub_fd:
            for ggh in unknown_ggh_set:
                if ggh[-1] == '\n' or ggh[-1] == '\r':
                    ub_fd.write('{0}'.format(ggh))
                else:
                    ub_fd.write('{0}\n'.format(ggh))

    @staticmethod
    def process_interm_txt_file(txt_file, ggh_set, unknown_ggh_set, bmy_set, ggh_to_bmy_dict):
        # 处理raw_domestic_brands.txt
        with open(txt_file, 'r', encoding='utf-8') as rdb_fd:
            for line in rdb_fd:
                arrs0 = line.split('*')
                ggh = arrs0[0]
                ggh_set.add(ggh)
                arrs1 = arrs0[-1][:-1].split('_')
                if len(arrs1) > 1:
                    bmy = '{0}_{1}_{2}'.format(arrs1[0], arrs1[1], arrs1[2])
                    bmy_set.add(bmy)
                    ggh_to_bmy_dict[ggh] = bmy
                else:
                    unknown_ggh_set.add(ggh)

    @staticmethod
    def process_domestic_folder(base_path: str, ggh_set, unknown_ggh_set, bmy_set, ggh_to_bmy_dict) -> int:
        base_obj = Path(base_path)
        for path_obj in base_obj.iterdir():
            full_name = str(path_obj)
            arrs0 = full_name.split('/')
            arrs1 = arrs0[-1].split('_')
            if len(arrs1) == 4:
                ggh = arrs1[-1]
                bmy = '{0}_{1}_{2}'.format(arrs1[0], arrs1[1], arrs1[2])
                ggh_set.add(ggh)
                if ggh in ggh_to_bmy_dict:
                    bmy1 = ggh_to_bmy_dict[ggh]
                    if bmy != bmy1:
                        print('冲突：{0}: {1} vs {2}'.format(ggh, bmy, bmy1))
                else:
                    ggh_to_bmy_dict[ggh] = bmy
            else:
                ggh = arrs0[-1]
                ggh_set.add(ggh)
                if not(ggh in ggh_to_bmy_dict):
                    unknown_ggh_set.add(ggh)
        return 18
        
    @staticmethod
    def merge_bmy_fgvc_id():
        '''
        将ggh_to_bmy_dict.txt中所有品牌_车型_年款与bmy_to_fgvc_id合并，
        按品牌-车型-年款排序后，写入bmy_to_fgvc_id_dict和fgvc_id_to_bmy_dict
        中
        '''
        bmy_set = set()
        with open('./work/ggh_to_bmy_dict.txt', 'r', encoding='utf-8') as gb_fd:
            for line in gb_fd:
                arrs0 = line.split(':')
                bmy = arrs0[1][:-2]
                bmy_set.add(bmy)
        with open('./work/bmy_to_fgvc_id_dict.txt', 'r', encoding='utf-8') as bf_fd:
            for line in bf_fd:
                arrs0 = line.split(':')
                arrs1 = arrs0[0].split('_')
                bmy = '{0}_{1}_{2}'.format(arrs1[0], arrs1[1], arrs1[2])
                bmy_set.add(bmy)
        bmy_set = sorted(bmy_set)
        fgvc_id = 0
        with open('./work/bmy_to_fgvc_id_dict.txt', 'w+', encoding='utf-8') as bfi_fd:
            with open('./work/fgvc_id_to_bmy_dict.txt', 'w+', encoding='utf-8') as fib_fd:
                for bmy in bmy_set:
                    print('bmy: {0}*{1};'.format(bmy, fgvc_id))
                    bfi_fd.write('{0}:{1}\n'.format(bmy, fgvc_id))
                    fib_fd.write('{0}:{1}\n'.format(fgvc_id, bmy))
                    fgvc_id += 1
        
    @staticmethod
    def generate_fgvc_ds():
        print('生成正式数据集')
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/train')
        bmy_to_fgvc_id_dict, fgvc_id_to_bmy_dict = DsManager.get_bmy_and_fgvc_id_dicts()
        with open('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/fgvc_train_ds_v2.txt', 'w+', encoding='utf-8') as ds_fd:
            for brand_obj in base_path.iterdir():
                brand_str = str(brand_obj)
                arrs0 = brand_str.split('/')
                brand_name = arrs0[-1]
                for model_obj in brand_obj.iterdir():
                    model_str = str(model_obj)
                    arrs1 = model_str.split('/')
                    model_name = arrs1[-1]
                    for year_obj in model_obj.iterdir():
                        year_str = str(year_obj)
                        arrs2 = year_str.split('/')
                        year_name = arrs2[-1]
                        bmy = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                        for file_obj in year_obj.iterdir():
                            print('{0}*{1}'.format(file_obj, bmy_to_fgvc_id_dict[bmy]))
                            ds_fd.write('{0}*{1}\n'.format(file_obj, bmy_to_fgvc_id_dict[bmy]))



