#
# 车管所测试工具类
#
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

class VaoTest(object):
    MODE_PREPARE_DATASET = 1001
    MODE_GET_VEHICLE_CODES = 1002
    # 从yt_train.txt中统计出所里附件的180个品牌中未覆盖的品牌
    MODE_GET_UNCOVERED_VCS = 1003
    # 找出未处理过的车辆编号
    MODE_GET_UNKNOWN_VCS = 1004
    # 生成92类已有类型数据集
    MODE_KNOWN_VCS_DS = 1005
    # 处理国产车2 guochanche_2
    MODE_PROCESS_GUOCHANCHE_2 = 1006
    # 生成现有品牌（86种）数据集
    MODE_KNOWN_86_DS = 1007
    # 绘制最新86种品牌数据集训练曲线
    MODE_B86_TRAIN_CURVE = 1008
    # 将测试数据集合并到训练数据集
    MODE_COPY_TEST_TO_TRAIN = 1009
    # 处理国产车2目录：guochanche_2
    MODE_PROCESS_GCC2 = 1010
    # 获取当前没有数据的品牌
    MODE_GET_UNCOVERED_BRANDS = 1011

    def __init__(self):
        self.name = 'util.VaoTest'

    @staticmethod
    def create_v_bn_no():
        for k, v in VaoTest.vehicle_brands.items():
            VaoTest.v_bn_no[v] = k

    @staticmethod
    def startup():
        mode = VaoTest.MODE_B86_TRAIN_CURVE
        if VaoTest.MODE_PREPARE_DATASET == mode:
            VaoTest.create_v_bn_no()
            print('新数据集生成')
            ds_file = './yt_train_d.txt'
            VaoTest.process_imported_vehicles_main(ds_file)
            VaoTest.process_domestic_vehicles_main(ds_file)
            # 统计每个类别图片数
            cids = {}
            for i in range(180):
                cids['{0}'.format(i)] = 0
            with open(ds_file, 'r', encoding='utf-8') as fd:
                for line in fd:
                    arrs = line.split('*')
                    class_id = arrs[-1][:-1]
                    cids[class_id] += 1
                    #print('class_id: {0};'.format(class_id))
            sum = 0
            for k, v in cids.items():
                if v > 0:
                    print('{0}: {1};'.format(k, v))
                    sum += 1
            print('总共品牌数：{0};'.format(sum))
            #VaoTest.process_test_data_main()
        elif VaoTest.MODE_GET_VEHICLE_CODES == mode:
            VaoTest.get_all_vehicle_codes()
        elif VaoTest.MODE_GET_UNCOVERED_VCS == mode:
            VaoTest.get_uncovered_vcs()
        elif VaoTest.MODE_GET_UNKNOWN_VCS == mode:
            VaoTest.get_unknown_vcs()
        elif VaoTest.MODE_KNOWN_VCS_DS == mode:
            VaoTest.known_vcs_ds_main()
        elif VaoTest.MODE_KNOWN_86_DS == mode:
            VaoTest.create_known_86_ds()
        elif VaoTest.MODE_B86_TRAIN_CURVE == mode:
            VaoTest.draw_b86_train_curve()
        elif VaoTest.MODE_COPY_TEST_TO_TRAIN == mode:
            VaoTest.copy_test_to_train()
        elif VaoTest.MODE_PROCESS_GCC2 == mode:
            VaoTest.process_gcc2()
        elif VaoTest.MODE_GET_UNCOVERED_BRANDS == mode:
            VaoTest.get_uncovered_brands()

    @staticmethod
    def get_all_vehicle_codes():
        '''
        从国产车guochanche_2目录中找出所有图片，取出车辆编号，
        形成不重复的车辆编号列表
        '''
        print('获取车辆型号列表...')
        base_dir = Path('/home/up/guochanche_2/')
        rst_file = './vehicle_codes.txt'
        vehicle_code_set = set()
        VaoTest.get_vehicle_codes_in_folder(base_dir, vehicle_code_set)
        print('编号总数量：{0};'.format(len(vehicle_code_set)))
        with open(rst_file, 'a+', encoding='utf-8') as fd:
            for vc in iter(vehicle_code_set):
                fd.write('{0}\r\n'.format(vc))

    @staticmethod
    def get_vehicle_codes_in_folder(folder_name, vehicle_code_set):
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(('jpg','png','jpeg','bmp')):
                arrs = full_name.split('/')
                arrs2 = arrs[-1].split('_')
                vehicle_code = arrs2[0]
                #rst_fd.write('{0}\r\n'.format(vehicle_code))
                vehicle_code_set.add(vehicle_code)
                print('vehicle_code: {0}; size={1};'.format(vehicle_code, len(vehicle_code_set)))
            elif file_obj.is_dir():
                VaoTest.get_vehicle_codes_in_folder(full_name, vehicle_code_set)
            else:
                print('ignore other file: {0};'.format(full_name))
        return vehicle_code_set

    @staticmethod
    def process_test_data_main():
        ds_file = './yt_test.txt'
        base_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/品牌')
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_test_vehicle_dataset(fd, base_dir)

    @staticmethod
    def create_test_vehicle_dataset(ds_fd, base_dir):
        path_obj = Path(base_dir)
        for file_obj in path_obj.iterdir():
            imgs_dir = str(file_obj)
            arrs = imgs_dir.split('/')
            last_seg = arrs[-1]
            class_id = int(last_seg[0:3]) - 1
            if class_id < 179:
                brand_name = VaoTest.vehicle_brands[last_seg[0:3]]
                VaoTest.list_img_files(ds_fd, imgs_dir, class_id)


    @staticmethod
    def process_domestic_vehicles_main(ds_file):
        base_dir = Path('/media/zjkj/My Passport/guochanche_all') #
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_domestic_vehicle_dataset(fd, base_dir)
    
    @staticmethod
    def create_domestic_vehicle_dataset(ds_fd, base_dir):
        vc_dict = {} # 车型编号和品牌字典
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as fd:
            line = fd.readline()
            while line:
                arrs = line.split('*')
                if len(arrs) > 1:
                    arrs2 = arrs[1].split('_')
                    brand_name = arrs2[0]
                    vc_dict[arrs[0]] = brand_name
                line = fd.readline()
        domestic_brands_items = [x for x in base_dir.iterdir() if base_dir.is_dir()]
        for item in domestic_brands_items:
            if not item.is_dir():
                continue
            item_str = str(item)
            arrs = item_str.split('/')
            brand_name = vc_dict[arrs[-1]] if arrs[-1] in vc_dict else '*'
            if '*' == brand_name:
                arrs1 = arrs[-1].split('_')
                if len(arrs1) > 1:
                    brand_name = arrs1[0]
            if brand_name in VaoTest.v_bn_no:
                class_id = int(VaoTest.v_bn_no[brand_name]) - 1
                VaoTest.list_img_files(ds_fd, item_str, class_id)

    @staticmethod
    def process_imported_vehicles_main(ds_file):
        base_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/vehicle_type_v2d/vehicle_type_v2d' #
        with open(ds_file, 'a+', encoding='utf-8') as fd:
            VaoTest.create_imported_vehicle_dataset(fd, base_dir)

    @staticmethod
    def create_imported_vehicle_dataset(ds_fd, folder_name):
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            if not file_obj.is_dir():
                continue
            imgs_dir = str(file_obj)
            arrs0 = imgs_dir.split('/')
            arrs = arrs0[-1].split('_')
            class_id = int(arrs[0]) - 1
            if class_id < 180:
                brand_name = arrs[1]
                VaoTest.list_img_files(ds_fd, imgs_dir, class_id)

    @staticmethod
    def list_img_files(ds_fd, folder_name, class_id):
        '''
        列出该目录以及其子目录下所有图片文件（以jpg为扩展名）列表
        '''
        print('processing: {0}...'.format(folder_name))
        path_obj = Path(folder_name)
        for file_obj in path_obj.iterdir():
            full_name = str(file_obj)
            if not file_obj.is_dir() and full_name.endswith(('jpg','png','jpeg','bmp')):
                #print('{0}*{1}'.format(file_obj, class_id))
                ds_fd.write('{0}*{1}\r\n'.format(file_obj, class_id))
            elif file_obj.is_dir():
                VaoTest.list_img_files(ds_fd, str(file_obj), class_id)
            else:
                print('ignore other file: {0};'.format(full_name))

    @staticmethod
    def process_vehicles():
        _, uncovered_brand_names = VaoTest.process_imported_vehicles()
        #uncovered_brand_names = []
        #for bn in VaoTest.vehicle_brands.values():
        #    uncovered_brand_names.append(bn)
        VaoTest.process_domestic_vehicles(uncovered_brand_names)

    @staticmethod
    def process_domestic_vehicles(uncovered_brand_names):
        print('处理国产车')
        vehicle_codes = []
        vc_dict = {} # 车型编号和品牌字典
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as fd:
            line = fd.readline()
            while line:
                print('# {0};'.format(line))
                arrs = line.split('*')
                print('len={0};'.format(len(arrs)))
                if len(arrs) > 1:
                    arrs2 = arrs[1].split('_')
                    brand_name = arrs2[0]
                    vc_dict[arrs[0]] = brand_name
                line = fd.readline()
        srcPath = Path('/media/zjkj/My Passport/guochanche_all')
        domestic_brands_items = [x for x in srcPath.iterdir() if srcPath.is_dir()]
        for item in domestic_brands_items:
            item_str = str(item)
            arrs = item_str.split('/')
            brand_name = vc_dict[arrs[-1]] if arrs[-1] in vc_dict else '*'
            if '*' == brand_name:
                arrs1 = arrs[-1].split('_')
                if len(arrs1) > 1:
                    brand_name = arrs1[0]
            print('domestic: {0} => {1};'.format(arrs[-1], brand_name))
            if brand_name in uncovered_brand_names:
                uncovered_brand_names.remove(brand_name)
        print('uncoder brand number: {0};'.format(len(uncovered_brand_names)))
        for bn in uncovered_brand_names:
            print(bn)

    @staticmethod
    def process_imported_vehicles():
        # 列出所有
        srcPath = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/vehicle_type_v2d/vehicle_type_v2d')
        imported_brands_items = [x for x in srcPath.iterdir() if srcPath.is_dir()]
        imported_brands_nos = []
        for item in imported_brands_items:
            item_str = str(item)
            arrs = item_str.split('_')
            arrs1 = arrs[-2].split('/')
            brand_no = arrs1[1]
            imported_brands_nos.append(brand_no)
            brand_name = arrs[-1]
            print('{0}: {1};'.format(brand_no, brand_name))
        # 从车型中去除进口车数据集中的品牌
        uncovered_brand_nos = []
        uncovered_brand_names = []
        for k in VaoTest.vehicle_brands.keys():
            if not (k in imported_brands_nos):
                uncovered_brand_nos.append(k)
                uncovered_brand_names.append(VaoTest.vehicle_brands[k])
        return uncovered_brand_nos, uncovered_brand_names
        # 从车型库中去除国产车数据集中的品牌
        # 打印未覆盖的车型


    @staticmethod
    def get_uncovered_vcs():
        our_vehicle_code_set = set()
        uncovered_vcs = []
        # 统计出已经处理完成的品牌
        train_ds = './yt_train.txt'
        with open(train_ds, 'r', encoding='utf-8') as fd:
            for line in fd:
                #print('正在处理：{0};'.format(line))
                arrs = line.split('*')
                class_id = arrs[1]
                our_vehicle_code_set.add(class_id[:-1])
        print('已经处理品牌数：{0}/180;'.format(len(our_vehicle_code_set)))
        for vc in our_vehicle_code_set:
            print(vc)
        # 找出未处理的品牌
        for k in VaoTest.vehicle_brands.keys():
            item = str(int(k)-1)
            rst = not (item in our_vehicle_code_set)
            print('{0} [{2}] => {1};'.format(k, rst, item))
            if rst:
                uncovered_vcs.append(k)
        print('未处理品牌数：{0};'.format(len(uncovered_vcs)))
        for vc in uncovered_vcs:
            print('##### {0};'.format(vc))

    @staticmethod
    def get_unknown_vcs():
        '''
        从./vehicle_codes.txt中挑出在./datasets/raw_domestic_brands.txt没有的编号
        '''
        # 获取所有的车辆编号
        all_vcs = set()
        with open('./vehicle_codes.txt', 'r', encoding='utf-8') as all_fd:
            for line in all_fd:
                all_vcs.add(line)
        print('所有车辆编号数量：{0};'.format(len(all_vcs)))
        # 获取已处理车辆编号
        known_vcs = set()
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as known_fd:
            for line in known_fd:
                arrs = line.split('*')
                known_vcs.add(arrs[0])
        print('已处理车辆编号数量：{0};'.format(len(known_vcs)))
        # 找出未处理车辆编号
        sum = 1
        for vc in known_vcs:
            if vc in all_vcs:
                all_vcs.remove(vc)
                print('从总体中删除：{0}; 共删除：{1};'.format(vc, sum))
                sum += 1
        with open('./unknown_vcs.txt', 'a+', encoding='utf-8') as unknown_fd:
            for vc in all_vcs:
                unknown_fd.write('{0}'.format(vc))

    @staticmethod
    def known_vcs_ds_main():
        #VaoTest.create_known_vcs_ds('./yt_train.txt', './datasets/CUB_200_2011/anno/yt_train_92.txt') # 生成训练数据集
        #VaoTest.create_known_vcs_ds('./datasets/CUB_200_2011/anno/yt_test.txt', './datasets/CUB_200_2011/anno/yt_test_92.txt')
        # 从0~91重复：找出yt_train_92.txt中所有符合条件的记录，
        # 从中随机取出1000张图片，形成yt_train_92_1000.txt作
        # 为训练数据集
        VaoTest.create_92_random_1000_ds()

    @staticmethod
    def create_known_vcs_ds(raw_file, ds_file):
        # 生成92类现有类别到0~91的对应表
        class_id = 0
        ob_nb_dict = {}
        nd_dict = {}
        with open('./work/known_brands.txt', 'r', encoding='utf-8') as kb_fd:
            for line in kb_fd:
                print('{0}=>{1};'.format(line[:-1], class_id))
                #on_fd.write('{0}={1}\n'.format(line[:-1], class_id))
                ob_nb_dict[line[:-1]] = class_id
                key = '{0:03d}'.format(int(line[:-1])+1)
                brand_name = VaoTest.vehicle_brands[key]
                #nd_fd.write('{0}={1}\n'.format('{0}'.format(class_id), brand_name))
                nd_dict['{0}'.format(class_id)] = brand_name
                class_id += 1
        sum = 0
        with open(raw_file, 'r', encoding='utf-8') as raw_fd:
            with open(ds_file, 'w+', encoding='utf-8') as ds92_fd:
                for line in raw_fd:
                    arrs = line.split('*')
                    img_file = arrs[0]
                    old_class_id = arrs[1][:-1]
                    if old_class_id in ob_nb_dict:
                        new_class_id = ob_nb_dict[old_class_id]
                        ds92_fd.write('{0}*{1}\n'.format(img_file, '{0}'.format(new_class_id)))
                        sum += 1
        print('共有{0}个样本！'.format(sum))

    @staticmethod
    def create_92_random_1000_ds():
        '''
        从0~91重复：找出yt_train_92.txt中所有符合条件的记录，
        从中随机取出1000张图片，形成yt_train_92_1000.txt
        作为训练数据集
        '''
        c_nums = {}
        fcid_set = set()
        for cid in range(92):
            c_nums['{0}'.format(cid)] = 0
        for cid in range(92):
            print('process {0} brand...'.format(cid))
            with open('./datasets/CUB_200_2011/anno/yt_train_92.txt', 'r', encoding='utf-8') as raw_fd:
                for line in raw_fd:
                    arrs = line.split('*')
                    fcid = int(arrs[1])
                    fcid_set.add(fcid)
                    img_file = arrs[0]
                    #print('fcid: {0};'.format(fcid))
                    if cid == fcid:
                        c_nums['{0}'.format(cid)] += 1
                        #print('get correct data')
        for k, v in c_nums.items():
            print('{0}: {1};'.format(k, v))
        print('fcid_set: {0};'.format(fcid_set))
        
    @staticmethod
    def create_known_86_ds():
        # 从原始数据集文件中读出记录，找到品牌编号
        # 若品牌编号在obn_nbn_dict的key里面，则将其写入新文件，并将其类别改为obn_nbn_dict中
        # 对应的新品牌编号
        VaoTest.create_know_86_ds_train_ds()
        #VaoTest.create_know_86_ds_test_ds()

    @staticmethod
    def create_know_86_ds_train_ds():
        raw_ds_file = './yt_train_d.txt'
        ds_file = './d2.txt'
        # 求出每个obn_nbn_dict的key的图片数量
        bn_nums = {}
        for k in VaoTest.obn_nbn_dict.keys():
            bn_nums[k] = 0
        with open(raw_ds_file, 'r', encoding='utf-8') as raw_fd:
            for line in raw_fd:
                arrs = line.split('*')
                brand_id = int(arrs[1][:-1])
                if brand_id in bn_nums:
                    bn_nums[brand_id] += 1
        #for k, v in bn_nums.items():
         #   print('b{0}: {1};'.format(k, v))
        with open(raw_ds_file, 'r', encoding='utf-8') as fd:
            with open(ds_file, 'w+', encoding='utf-8') as ds_fd:
                for line in fd:
                    arrs = line.split('*')
                    img_file = arrs[0]
                    brand_id = int(arrs[1][:-1])
                    if brand_id in VaoTest.obn_nbn_dict:
                        new_brand_id = VaoTest.obn_nbn_dict[brand_id]
                        #print('brand_id: {0} => {1}*{2};'.format(brand_id, img_file, new_brand_id))
                        write_to_file = False
                        if bn_nums[brand_id] < 1000:
                            write_to_file = True
                        elif random.random() < 1000.0 / bn_nums[brand_id]:
                            write_to_file = True
                        if write_to_file:
                            ds_fd.write('{0}*{1}\n'.format(img_file, new_brand_id))
        nbn_nums = {}
        for k in range(85):
            nbn_nums[k] = 0
        with open(ds_file, 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs = line.split('*')
                brand_id = int(arrs[1][:-1])
                if brand_id in nbn_nums:
                    nbn_nums[brand_id] += 1
        for k, v in nbn_nums.items():
            print('new {0}: {1};'.format(k, v))



    @staticmethod
    def create_know_86_ds_test_ds():
        raw_ds_file = './datasets/CUB_200_2011/anno/yt_test.txt'
        ds_file = './d1.txt'
        with open(raw_ds_file, 'r', encoding='utf-8') as fd:
            with open(ds_file, 'w+', encoding='utf-8') as ds_fd:
                for line in fd:
                    arrs = line.split('*')
                    img_file = arrs[0]
                    brand_id = int(arrs[1][:-1])
                    if brand_id in VaoTest.obn_nbn_dict:
                        new_brand_id = VaoTest.obn_nbn_dict[brand_id]
                        #print('brand_id: {0} => {1}*{2};'.format(brand_id, img_file, new_brand_id))
                        ds_fd.write('{0}*{1}\n'.format(img_file, new_brand_id))

    @staticmethod
    def draw_b86_train_curve():
        '''
        从训练记录文件中读出训练步数和Top1精度，使用matplotlib绘制
        '''
        src_path = Path('./net_model/training_descibe_5910_CUB')
        chpts = [x for x in src_path.iterdir() if src_path.is_dir()]
        X0 = []
        acc1 = {}
        step_per_epoch = 2705
        for chpt in chpts:
            print(chpt)
            arrs0 = str(chpt).split('/')
            item = arrs0[-1]
            arrs = item.split('_')
            epoch = int(arrs[1])
            step = int(arrs[2])
            total_step = epoch*step_per_epoch + step
            X0.append(total_step)
            acc1[total_step] = float(arrs[3])
        # 将X0进行排序
        print('before sort: {0};'.format(X0))
        X0.sort()
        print('after sort: {0};'.format(X0))
        y = []
        for step in X0:
            y.append(acc1[step])
            print('{0} = {1};'.format(step, acc1[step]))
        fig, ax = plt.subplots()
        ax.plot(np.array(X0, dtype=np.float32), np.array(y, dtype=np.float32)*100.0, label='top1')
        ax.set_xlabel('steps')
        ax.set_ylabel('accuracy')
        ax.set_title("accuracy curve")
        ax.legend()
        plt.show()

    @staticmethod
    def copy_test_to_train():
        '''
        将测试数据集数据拷贝到训练数据集中，用于测试能够在到的精度
        '''
        src_file = './datasets/CUB_200_2011/anno/yt_test.txt'
        dst_file = './datasets/CUB_200_2011/anno/yt_train.txt'
        with open(src_file, 'r', encoding='utf-8') as src_fd:
            with open(dst_file, 'a+', encoding='utf-8') as dst_fd:
                for line in src_fd:
                    dst_fd.write(line)

    @staticmethod
    def process_gcc2():
        '''
        处理国产车2目录下内容
        '''
        # 找出在本目录下出现并在附件2列表中，但是未在已处理的86类中的品牌
        vc_file = './work/gcc2_vc_bmy.txt'
        with open(vc_file, 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs = line.split('*')
                vc = arrs[0]
                if len(arrs) > 1:
                    bmy = arrs[1][:-1]
                    arrs2 = bmy.split('_')
                    brand_name = arrs2[0]
                    print('{0} <=> {1};  brand: {2};'.format(vc, bmy, brand_name))

    @staticmethod
    def get_uncovered_brands():
        '''
        获取在vehicle_type_v2d、guochanche_all、guochanche_2中没有，但是却在所里附件2中存在的品牌
        '''
        i_debug = 10
        sum = 0
        unfind_vcs = './unfind_vcs.txt'
        with open('./datasets/raw_domestic_brands.txt', 'r', encoding='utf-8') as fd:
            with open(unfind_vcs, 'w+', encoding='utf-8') as uv_fd:
                for line in fd:
                    arrs = line.split('*')
                    if len(arrs) <= 1:
                        print(arrs[0][:-1])
                        uv_fd.write('{0}\n'.format(arrs[0][:-1]))
                        sum += 1
        print('共有{0}条'.format(sum))
        if 1 == i_debug:
            return
        # 处理进口车
        uncovered_brands = []
        VaoTest.create_v_bn_no()
        for k, v in VaoTest.vehicle_brands.items():
            uncovered_brands.append(k)
        print(VaoTest.v_bn_no)
        print(uncovered_brands)
        # 处理进口车
        iv_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/vehicle_type_v2d/vehicle_type_v2d')
        for file_obj in iv_dir.iterdir():
            file_name = str(file_obj)
            arrs0 = file_name.split('/')
            arrs1 = arrs0[-1].split('_')
            brand_no = arrs1[0]
            if brand_no in uncovered_brands:
                print('删除进口车编号为{0}的品牌'.format(brand_no))
                uncovered_brands.remove(brand_no)
        print('阶段1：未处理品牌数={0};'.format(len(uncovered_brands)))
        # 处理国产车
        dv_file = './datasets/raw_domestic_brands.txt'
        with open(dv_file, 'r', encoding='utf-8') as dv_fd:
            for line in dv_fd:
                arrs0 = line.split('*')
                if len(arrs0) > 1:
                    arrs1 = arrs0[1].split('_')
                    brand_name = arrs1[0]
                    if brand_name in VaoTest.v_bn_no.keys():
                        brand_no = VaoTest.v_bn_no[brand_name]
                        if brand_no in uncovered_brands:
                            uncovered_brands.remove(brand_no)
                            print('删除国产国编号为{0}的品牌'.format(brand_no))
        print('阶段2：未处理品牌数={0};'.format(len(uncovered_brands)))
        print('未处理品牌列表：')
        for bno in uncovered_brands:
            print('{0} = {1};'.format(bno, VaoTest.vehicle_brands[bno]))
        




    
    v_no_bn = {}
    v_bn_no = {}
    v_no_set = set()
    v_bn_set = set()
    vehicle_brands = {
        '001':'奥迪',
        '002':'阿尔法罗密欧',
        '003':'阿斯顿马丁',
        '004':'奔驰',
        '005':'宝马',
        '006':'宾利',
        '007':'布嘉迪',
        '008':'保时捷',
        '009':'别克',
        '010':'本田',
        '011':'标致',
        '012':'比亚迪',
        '013':'北京汽车',
        '014':'宝骏',
        '015':'宝腾',
        '016':'长城',
        '017':'长安',
        '018':'长丰',
        '019':'昌河',
        '020':'川汽野马',
        '021':'东风',
        '022':'风神',
        '023':'大发',
        '024':'帝豪',
        '025':'东南',
        '026':'道奇',
        '027':'大众',
        '028':'大宇',
        '029':'大迪',
        '030':'法拉利',
        '031':'丰田',
        '032':'皇冠',
        '033':'福特',
        '034':'野马',
        '035':'菲亚特',
        '036':'福田',
        '037':'福迪',
        '038':'富奇',
        '039':'广汽',
        '040':'GMC',
        '041':'光冈',
        '042':'海马',
        '043':'哈飞',
        '044':'悍马',
        '045':'霍顿',
        '046':'华普',
        '047':'华泰',
        '048':'红旗',
        '049':'黄海',
        '050':'汇众',
        '051':'捷豹',
        '052':'吉普',
        '053':'金杯',
        '054':'江淮',
        '055':'吉利',
        '056':'江铃',
        '057':'江南',
        '058':'吉奥',
        '059':'解放',
        '060':'九龙',
        '061':'金龙',
        '062':'凯迪拉克',
        '063':'克莱斯勒',
        '064':'柯尼塞格',
        '065':'开瑞',
        '066':'KTM',
        '067':'克尔维特',
        '068':'兰博基尼',
        '069':'劳斯莱斯',
        '070':'路虎',
        '071':'莲花',
        '072':'林肯',
        '073':'雷克萨斯',
        '074':'铃木',
        '075':'雷诺',
        '076':'力帆',
        '077':'陆风',
        '078':'理念',
        '079':'迈巴赫',
        '080':'名爵',
        '081':'迷你',
        '082':'玛莎拉蒂',
        '083':'马自达',
        '084':'纳智捷',
        '085':'南汽',
        '086':'欧宝',
        '087':'讴歌',
        '088':'奥兹莫比尔',
        '089':'帕加尼',
        '090':'庞蒂克',
        '091':'奇瑞',
        '092':'起亚',
        '093':'全球鹰',
        '094':'庆铃',
        '095':'启辰',
        '096':'尼桑',
        '097':'瑞麒',
        '098':'荣威',
        '099':'罗森',
        '100':'罗孚',
        '101':'萨博',
        '102':'斯巴鲁',
        '103':'双环',
        '104':'世爵',
        '105':'斯派朗',
        '106':'三菱',
        '107':'双龙',
        '108':'smart',
        '109':'斯柯达',
        '110':'塔塔',
        '111':'土星',
        '112':'沃尔沃',
        '113':'威麟',
        '114':'五菱',
        '115':'威兹曼',
        '116':'沃克斯豪尔',
        '117':'五十铃',
        '118':'现代',
        '119':'雪佛兰',
        '120':'夏利',
        '121':'雪铁龙',
        '122':'西亚特',
        '123':'英菲尼迪',
        '124':'英伦',
        '125':'一汽',
        '126':'奔腾',
        '127':'跃进',
        '128':'依维柯',
        '129':'永源',
        '130':'中华',
        '131':'众泰',
        '132':'中兴',
        '133':'中顺',
        '134':'华阳',
        '135':'飞虎',
        '136':'北汽幻速',
        '137':'北汽威望',
        '138':'北汽制造',
        '139':'安凯',
        '140':'北奔重汽',
        '141':'大运',
        '142':'东沃（优迪）',
        '143':'福达',
        '144':'日野',
        '145':'海格',
        '146':'红岩',
        '147':'华菱星马（CAMC）',
        '148':'金旅',
        '149':'凯马',
        '150':'东风柳汽',
        '151':'青年',
        '152':'三环',
        '153':'三一重工',
        '154':'陕汽重卡',
        '155':'少林',
        '156':'申龙',
        '157':'时代（福田）',
        '158':'时风',
        '159':'斯堪尼亚（SCANIA）',
        '160':'唐骏',
        '161':'五征',
        '162':'徐工汽车',
        '163':'亚星',
        '164':'英田',
        '165':'宇通',
        '166':'中国重汽',
        '167':'重汽王牌',
        '168':'中通',
        '169':'观致',
        '170':'特斯拉',
        '171':'福汽启腾',
        '172':'哈弗',
        '173':'江铃轻汽',
        '174':'江铃驭胜',
        '175':'上汽大通',
        '176':'思铭',
        '177':'腾势',
        '178':'英致',
        '179':'凯翼',
        '180':'北方客车'
        }

    obn_nbn_dict = {
        0: 0,
        1: 1,
        3: 2,
        4: 3,
        5: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8,
        11: 9,
        13: 10,
        15: 11,
        16: 12,
        18: 13,
        20: 14,
        21: 15,
        23: 16,
        24: 17,
        25: 18,
        26: 19,
        30: 20,
        32: 21,
        33: 22,
        34: 23,
        35: 24,
        36: 25,
        41: 26,
        42: 27,
        45: 28,
        46: 29,
        47: 30,
        48: 31,
        49: 32,
        50: 33,
        51: 34,
        52: 35,
        53: 36,
        54: 37,
        55: 38,
        56: 39,
        57: 40,
        59: 41,
        61: 42,
        62: 43,
        64: 44,
        69: 45,
        70: 46,
        71: 47,
        72: 48,
        73: 49,
        74: 50,
        75: 51,
        76: 52,
        77: 53,
        81: 54,
        82: 55,
        83: 56,
        85: 57,
        86: 58,
        90: 59,
        92: 60, 
        93: 61,
        94: 62,
        96: 63,
        97: 64,
        101: 65,
        102: 66,
        105: 67,
        108: 68,
        111: 69,
        112: 70,
        113: 71,
        116: 72,
        117: 73,
        118: 74,
        119: 75,
        120: 76,
        122: 77,
        123: 78,
        124: 79,
        125: 80,
        127: 81,
        128: 82,
        129: 83,
        130: 84
    }
    
    nbn_name_dict = {
        0: '奥迪',
        1: '阿尔法罗密欧',
        2: '奔驰',
        3: '宝马',
        4: '宾利',
        5: '保时捷',
        6: '别克',
        7: '本田',
        8: '标致',
        9: '比亚迪',
        10: '宝骏',
        11: '长城',
        12: '长安',
        13: '昌河',
        14: '东风',
        15: '风神',
        16: '帝豪',
        17: '东南',
        18: '道奇',
        19: '大众',
        20: '丰田',
        21: '福特',
        22: '野马',
        23: '菲亚特',
        24: '福田',
        25: '福迪',
        26: '海马',
        27: '哈飞',
        28: '华普',
        29: '华泰',
        30: '红旗',
        31: '黄海',
        32: '汇众',
        33: '捷豹',
        34: '吉普',
        35: '金杯',
        36: '江淮',
        37: '吉利',
        38: '江铃',
        39: '江南',
        40: '吉奥',
        41: '九龙',
        42: '凯迪拉克',
        43: '克莱斯勒',
        44: '开瑞',
        45: '路虎',
        46: '莲花',
        47: '林肯',
        48: '雷克萨斯',
        49: '铃木',
        50: '雷诺',
        51: '力帆',
        52: '陆风',
        53: '理念',
        54: '玛莎拉蒂',
        55: '马自达',
        56: '纳智捷',
        57: '欧宝',
        58: '讴歌',
        59: '奇瑞',
        60: '全球鹰',
        61: '庆铃',
        62: '启辰',
        63: '瑞麒',
        64: '荣威',
        65: '斯巴鲁',
        66: '双环',
        67: '三菱',
        68: '斯柯达',
        69: '沃尔沃',
        70: '威麟',
        71: '五菱',
        72: '五十铃',
        73: '现代',
        74: '雪佛兰',
        75: '夏利',
        76: '雪铁龙',
        77: '英菲尼迪',
        78: '英伦',
        79: '一汽',
        80: '奔腾',
        81: '依维柯',
        82: '永源',
        83: '中华',
        84: '众泰'
    }












