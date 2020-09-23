# 由于在车辆检测时，会将所有检测json保存在同一目录下，会造成
# 文件读写缓慢的问题，所以需要一个独立的线程，将Json文件定期
# 从该目录中拷贝出来，存储到一个按文件编号为层次结构的规整目录
# 下面。
import os
import json
import requests
from pathlib import Path
import threading
import cv2
import numpy as np
#
from apps.wxs.fu.file_tree_folder_saver import FileTreeFolderSaver
from apps.wxs.fu.file_util import FileUtil

class VdJsonManager(object):
    # 车头车尾类型
    HTT_HEAD = 'head'
    HTT_TAIL = 'tail'
    # 车辆类型
    VT_CAR = 'car'
    VT_TRUCK = 'truck'
    VT_BUS = 'bus'
    # 运行模式定义
    RM_SAVE_JSONS_IN_TREE_FOLDER = 1
    RM_PARSE_VD_JSON = 2
    RM_GET_RAW_IMG_FILE_TO_FULL_FN = 3
    # 多线程同步锁
    s_lock = None
    
    def __init__(self):
        self.refl = 'apps.wxs.vdc.VdJsonManager'
        
    def start(self):
        mode = VdJsonManager.RM_PARSE_VD_JSON
        if VdJsonManager.RM_SAVE_JSONS_IN_TREE_FOLDER == mode:
            VdJsonManager.save_jsons_in_tree_folder()
        elif VdJsonManager.RM_PARSE_VD_JSON == mode:
            VdJsonManager.parse_vd_jsons()
        # ***********************************************************************************
        # ***********************************************************************************
        elif VdJsonManager.RM_GET_RAW_IMG_FILE_TO_FULL_FN == mode:
            '''
            获取fgvc_dataset/raw目录下图片文件名与全路径文件名对应关系
            '''
            img_file_to_full_fn = VdJsonManager.get_raw_img_file_to_full_fn()
            num = 0
            for k, v in img_file_to_full_fn.items():
                print('{0} => {1};'.format(k, v))
                num += 1
                if num > 20:
                    break
        else:
            print('unknow mode')
    
    @staticmethod
    def save_jsons_in_tree_folder():
        '''
        将车辆检测Json文件从一个目录拷贝到树形目录下
        '''
        # 将Json文件以规整的目录格式存放
        #base_path = Path('/media/zjkj/work/fgvc_dataset/raw_json')
        base_folder = '/media/zjkj/work/fgvc_dataset/raw_json'
        dst_folder = '/media/zjkj/work/fgvc_dataset/vdc0907/json_500'
        file_id = 1229700
        num = 0
        with open('/media/zjkj/work/fgvc_dataset/vdcj.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                full_fn = '{0}/{1}'.format(base_folder, line)
                if os.path.exists(full_fn):
                    file_id = FileTreeFolderSaver.save_file(dst_folder, full_fn, file_id)
                    if file_id % 100 == 0:
                        print('处理完成{0}个文件'.format(file_id))
                else:
                    num += 1
                    if num % 100 == 0:
                        print('预处理{0}个文件'.format(num))
                
        '''
        for jf_obj in base_path.iterdir():
            full_fn = str(jf_obj)
            file_id = FileTreeFolderSaver.save_file(dst_folder, full_fn, file_id)
            if file_id % 100 == 0:
                print('处理完成{0}个文件'.format(file_id))
        '''
    
    s_num = 0
    @staticmethod
    def parse_vd_jsons():
        img_file_to_full_fn = VdJsonManager.get_raw_img_file_to_full_fn()
        print('完成获取图片文件名到全路径文件名字典')
        cutted_images = VdJsonManager.get_cutted_images()
        print('完成获取切图完成的图片文件名集合')
        VdJsonManager.s_lock = threading.RLock()
        '''
        params = {'idx': 0, 'iffn_dict': img_file_to_full_fn, 'cutted_images': cutted_images}
        thd = threading.Thread(target=VdJsonManager.process_vd_json_thd, args=(params,))
        thd.start()
        thd.join()
        '''
        thds = []
        for idx in range(11):
            params = {'idx': idx, 'iffn_dict': img_file_to_full_fn, 'cutted_images': cutted_images}
            thd = threading.Thread(target=VdJsonManager.process_vd_json_thd, args=(params,))
            thds.append(thd)
        for thd in thds:
            thd.start()
        for thd in thds:
            thd.join()
        
    @staticmethod
    def get_img_file_in_vd_jf(json_full_fn):
        '''
        从车辆检测json文件中取出图片文件名称
        '''
        arrs_a = json_full_fn.split('/')
        fn = arrs_a[-1]
        arrs_b = fn.split('_')
        return '{0}_{1}_{2}_{3}_{4}'.format(arrs_b[0], arrs_b[1], arrs_b[2], arrs_b[3], arrs_b[4])
        
    @staticmethod
    def get_cutted_images():
        '''
        求出已经切图的图片文件列表
        '''
        cutted_images = set()
        base_path = Path('./support/datasets/train')
        for ht_obj in base_path.iterdir():
            for bct_obj in ht_obj.iterdir():
                for sf1 in bct_obj.iterdir():
                    for sf2 in sf1.iterdir():
                        for sf3 in sf2.iterdir():
                            for sf4 in sf3.iterdir():
                                for sf5 in sf4.iterdir():
                                    for file_obj in sf5.iterdir():
                                        full_fn = str(file_obj)
                                        if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg')):
                                            arrs_a = full_fn.split('/')
                                            img_file = arrs_a[-1]
                                            cutted_images.add(img_file)
        return cutted_images
        
    @staticmethod
    def process_vd_json_thd(params):
        cut_img_head_folder = './support/datasets/train'
        cut_img_tail_folder = './support/datasets/train'
        img_file_to_full_fn = params['iffn_dict']
        idx = params['idx']
        cutted_images = params['cutted_images']
        num = 0
        cars = ['13', '14']
        trucks = ['21', '22']
        buss = ['11', '12']
        miss_images_fd = open('./support/miss_images.txt', 'w+', encoding='utf-8')
        with open('./support/vd_error_jsons.txt', 'w+', encoding='utf-8') as efd:
            with open('./support/vd_jsons_{0:02d}.txt'.format(idx), 'r', encoding='utf-8') as vfd:
                for line in vfd:
                    line = line.strip()
                    full_fn = line
                    j_img_file = VdJsonManager.get_img_file_in_vd_jf(full_fn)
                    if j_img_file in cutted_images:
                        VdJsonManager.s_num += 1
                        if VdJsonManager.s_num % 1000 == 0:
                            print('### Thread_{0}: cut and save {1};'.format(idx, VdJsonManager.s_num))
                        continue
                    psfx, cllxfl, xlwz = VdJsonManager.parse_vd_json(full_fn)
                    if psfx is not None:
                        arrs_a = full_fn.split('/')
                        json_fn = arrs_a[-1]
                        arrs_b = json_fn.split('_')
                        img_file = '{0}_{1}_{2}_{3}_{4}'.format(
                            arrs_b[0], arrs_b[1], arrs_b[2],
                            arrs_b[3], arrs_b[4]
                        )
                        head_tail = VdJsonManager.HTT_HEAD
                        if psfx == '2':
                            head_tail = VdJsonManager.HTT_TAIL
                        vehicle_type = VdJsonManager.VT_CAR
                        if cllxfl in trucks:
                            vehicle_type = VdJsonManager.VT_TRUCK
                        elif cllxfl in buss:
                            vehicle_type = VdJsonManager.VT_BUS
                        if img_file in img_file_to_full_fn:
                            img_full_fn = img_file_to_full_fn[img_file]
                        else:
                            print('     missing image file: {0}'.format(img_file))
                            miss_images_fd.write('{0}\n'.format(img_file))
                            continue
                        arrs_c = xlwz.split(',')
                        box = [int(arrs_c[0]), int(arrs_c[1]), int(arrs_c[2]), int(arrs_c[3])]
                        if box[0] < 0:
                            box[0] = 0
                        if box[1] < 0:
                            box[1] = 0
                        try:
                            VdJsonManager.s_lock.acquire() # 获取锁以进行目录操作
                            VdJsonManager.s_num, dst_cut_fn = FileTreeFolderSaver.get_dst_fn('{0}/{1}/{2}'.format(cut_img_head_folder, head_tail, vehicle_type), img_full_fn, VdJsonManager.s_num)
                            croped_img = VdJsonManager.crop_and_resize_img(img_full_fn, box)
                            cv2.imwrite(dst_cut_fn, croped_img)
                            VdJsonManager.s_lock.release()
                        except Exception as ex:
                            print('##### Exception {0};'.format(ex))
                            VdJsonManager.s_lock.release()
                        if VdJsonManager.s_num % 1000 == 0:
                            print('Thread_{0}: cut and save {1};'.format(idx, VdJsonManager.s_num))
                    else:
                        efd.write('{0}\n'.format(full_fn))
        miss_images_fd.close()

    @staticmethod
    def crop_and_resize_img(img_file, box, size=(224, 224), mode=1):
        if mode == 1:
            return VdJsonManager.crop_and_resize_no_aspect(img_file, box, size)
        else:
            return VdJsonManager.crop_and_resize_keep_aspect(img_file, box, size)

    @staticmethod
    def crop_and_resize_no_aspect(img_file, box, size=(224, 224), mode=1):
        org_img = cv2.imread(img_file)
        crop_img = org_img[
            box[1] : box[1] + box[3],
            box[0] : box[0] + box[2]
        ]
        '''
        plt.subplot(1, 3, 1)
        plt.title('org_img: {0}*{1}'.format(org_img.shape[0], org_img.shape[1]))
        plt.imshow(org_img)
        plt.subplot(1, 3, 2)
        plt.title('img: {0}*{1}'.format(crop_img.shape[0], crop_img.shape[1]))
        plt.imshow(crop_img)
        resized_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_LINEAR)
        plt.subplot(1, 3, 3)
        plt.title('resized')
        plt.imshow(resized_img)
        plt.show()
        '''
        return crop_img

    @staticmethod
    def crop_and_resize_keep_aspect(img_file, box, size=(224, 224), mode=1):
        pass
        
    @staticmethod
    def save_vd_json_fns():
        def process_json_files(fds):     
            base_path = Path('/media/zjkj/work/fgvc_dataset/vdc0907/json_500')                                       
            num = 0
            for sf1 in base_path.iterdir():
                for sf2 in sf1.iterdir():
                    for sf3 in sf2.iterdir():
                        for sf4 in sf3.iterdir():
                            for sf5 in sf4.iterdir():
                                for jf_obj in sf5.iterdir():
                                    full_fn = str(jf_obj)
                                    idx = num // 600000
                                    fds[idx].write('{0}\n'.format(full_fn))
                                    num += 1
                                    if num % 100 == 0:
                                        print('记录{0}个Json文件'.format(num))
        fds = []
        with open('./support/vd_jsons_00.txt', 'w+', encoding='utf-8') as fd0:
            fds.append(fd0)
            with open('./support/vd_jsons_01.txt', 'w+', encoding='utf-8') as fd1:
                fds.append(fd1)
                with open('./support/vd_jsons_02.txt', 'w+', encoding='utf-8') as fd2:
                    fds.append(fd2)
                    with open('./support/vd_jsons_03.txt', 'w+', encoding='utf-8') as fd3:
                        fds.append(fd3)
                        with open('./support/vd_jsons_04.txt', 'w+', encoding='utf-8') as fd4:
                            fds.append(fd4)
                            with open('./support/vd_jsons_05.txt', 'w+', encoding='utf-8') as fd5:
                                fds.append(fd5)
                                with open('./support/vd_jsons_06.txt', 'w+', encoding='utf-8') as fd6:
                                    fds.append(fd6)
                                    with open('./support/vd_jsons_07.txt', 'w+', encoding='utf-8') as fd7:
                                        fds.append(fd7)
                                        with open('./support/vd_jsons_08.txt', 'w+', encoding='utf-8') as fd8:
                                            fds.append(fd8)
                                            with open('./support/vd_jsons_09.txt', 'w+', encoding='utf-8') as fd9:
                                                fds.append(fd9)
                                                with open('./support/vd_jsons_10.txt', 'w+', encoding='utf-8') as fd10:
                                                    fds.append(fd10)
                                                    process_json_files(fds)
        
    @staticmethod
    def parse_vd_json(json_file):
        cllxfls = ['11', '12', '13', '14', '21', '22']
        with open(json_file, 'r', encoding='utf-8') as jfd:
            data = json.load(jfd)
            return VdJsonManager.parse_vd_json_data(data)
                                
                                
    @staticmethod
    def get_raw_img_file_to_full_fn():
        img_file_to_full_fn = {}
        iffn_file = './support/raw_iffn.txt'
        #base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
        base_folder = '/media/zjkj/work/fgvc_dataset/raw'
        num = 0
        if os.path.exists(iffn_file):
            print('从缓存文件中读取...')
            with open(iffn_file, 'r', encoding='utf-8') as rfd:
                for line in rfd:
                    line = line.strip()
                    arrs_a = line.split(':')
                    img_file = arrs_a[0]
                    full_fn = arrs_a[1]
                    img_file_to_full_fn[img_file] = full_fn
        else:
            print('从文件目录中读取...')
            img_file_to_full_fn = FileUtil.get_files_in_subfolders_dict(base_folder)
            '''
            for brand_obj in base_path.iterdir():
                for bm_obj in brand_obj.iterdir():
                    for bmy_obj in bm_obj.iterdir():
                        for file_obj in bmy_obj.iterdir():
                            full_fn = str(file_obj)
                            if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                                arrs_a = full_fn.split('/')
                                img_file = arrs_a[-1]
                                img_file_to_full_fn[img_file] = full_fn
                                num += 1
                                if num % 100 == 0:
                                    print('加入字典文件数：{0};'.format(num))
                            elif file_obj.is_dir():
                                for sf1 in file_obj.iterdir():
                                    if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                                        arrs_a = full_fn.split('/')
                                        img_file = arrs_a[-1]
                                        img_file_to_full_fn[img_file] = full_fn
                                        num += 1
                                        if num % 100 == 0:
                                            print('加入字典文件数：{0};'.format(num))
            '''
            with open(iffn_file, 'w+', encoding='utf-8') as wfd:
                for k, v in img_file_to_full_fn.items():
                    wfd.write('{0}:{1}\n'.format(k, v))
        return img_file_to_full_fn
        
    @staticmethod
    def run_vd_cut_save():
        print('利用Python程序完成整个切图流程')
        # 统计已经完成切图的图片文件集合
        ci_num = 0
        cutted_image_set = set()
        for root, dirs, files in os.walk('/media/zjkj/work/yantao/zjkj/i900m_cutted', topdown=False):
        #for root, dirs, files in os.walk('/media/zjkj/work/yantao/zjkj/raw_cutted', topdown=False):
            for fn in files:
                cutted_image_set.add(fn)
                ci_num += 1
                if ci_num % 1000 == 0:
                    print('统计完成切图文件：{0}个'.format(ci_num))
        print('已经切图数量：{0};'.format(len(cutted_image_set)))            
        txts_num = 20
        VdJsonManager.s_num = 0
        VdJsonManager.s_lock = threading.RLock()
        miss_images_fd = open('./support/m900_miss_images.txt', 'w+', encoding='utf-8')
        #miss_images_fd = open('./support/raw_miss_images.txt', 'w+', encoding='utf-8')
        efd = open('./support/m900_error.txt', 'w+', encoding='utf-8')
        #efd = open('./support/raw_error.txt', 'w+', encoding='utf-8')
        # 将图片文件列表均匀分给20个文本文件
        ifds = []
        thds = []
        for idx in range(txts_num):
            #fd = open('./support/raw_{0:02d}.txt'.format(idx), 'r', encoding='utf-8')
            fd = open('./support/i900m_{0:02d}.txt'.format(idx), 'r', encoding='utf-8')
            ifds.append(fd)
            params = params = {
                'idx': idx , 'fd': fd, 'efd': efd, 
                'miss_images_fd': miss_images_fd,
                'cutted_image_set': cutted_image_set,
                'cut_img_folder': '/media/zjkj/work/yantao/zjkj/i900m_cutted'
            }
            print('idx={0}: {1};'.format(idx, len(ifds)))
            thd = threading.Thread(target=VdJsonManager.vd_cut_save_thd, args=(params,))
            thds.append(thd)
        
        '''
        for idx in range(txts_num):
            fd = open('./support/i900m_{0:02d}.txt'.format(idx), 'w+', encoding='utf-8')
            ifds.append(fd)
        VdJsonManager.get_image_full_fns_to_txts(txts_num, ifds)
        
        for ifd in ifds:
            ifd.close()
        miss_images_fd.close()
        efd.close()
        i_debug = 1
        if 1 == i_debug:
            return
        '''
        # 等待线程池结束
        for thd in thds:
            thd.start()
        for thd in thds:
            thd.join()
        # 关闭文件
        for ifd in ifds:
            ifd.close()
        miss_images_fd.close()
        efd.close()
            
        
        
    @staticmethod
    def get_image_full_fns_to_txts(txts_num, ifds):
        num = 0
        # 进口车
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
        for brand_obj in base_path.iterdir():
            for bm_obj in brand_obj.iterdir():
                for bmy_obj in bm_obj.iterdir():
                    for file_obj in bmy_obj.iterdir():
                        full_fn = str(file_obj)
                        if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg')):
                            num += 1
                            ifds[num % txts_num].write('{0}\n'.format(full_fn))
                            if num % 1000 == 0:
                                print('进口车获取到{0}个文件'.format(num))
        print('共有{0}个文件'.format(num))
        '''
        # 国产车
        base_path = Path('/media/ps/My/总已完成')
        #img_full_fns = []
        for sf1 in base_path.iterdir():
            for vc_obj in sf1.iterdir():
                for file_obj in vc_obj.iterdir():
                    full_fn = str(file_obj)
                    if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg')):
                        #img_full_fns.append(full_fn)
                        num += 1
                        ifds[num % txts_num].write('{0}\n'.format(full_fn))
                        if num % 1000 == 0:
                            print('获取到{0}个文件'.format(num))
        print('共有{0}个文件'.format(num))
        '''
        
    @staticmethod
    def vd_cut_save_thd(params):
        #cut_img_folder = '/media/zjkj/work/yantao/zjkj/raw_cutted'
        #cut_img_folder = '/media/zjkj/work/yantao/zjkj/i900m_cutted'
        cars = ['13', '14']
        trucks = ['21', '22']
        buss = ['11', '12']
        idx = params['idx']
        fd = params['fd']
        miss_images_fd = params['miss_images_fd']
        efd = params['efd']
        cutted_image_set = params['cutted_image_set']
        cut_img_folder = params['cut_img_folder']
        for line in fd:
            line = line.strip()
            full_fn = line
            arrs_a = full_fn.split('/')
            img_file = arrs_a[-1]
            if img_file in cutted_image_set:
                # 忽略已经完成切图的文件
                VdJsonManager.s_num += 1
                if VdJsonManager.s_num % 1000 == 0:
                    print('已经处理完成{0}个文件'.format(VdJsonManager.s_num))
                continue
            data = VdJsonManager.get_img_reid_feature_vector(full_fn)
            psfx, cllxfl, xlwz = VdJsonManager.parse_vd_json_data(data)
            if psfx is not None:
                head_tail = VdJsonManager.HTT_HEAD
                if psfx == '2':
                    head_tail = VdJsonManager.HTT_TAIL
                vehicle_type = VdJsonManager.VT_CAR
                if cllxfl in trucks:
                    vehicle_type = VdJsonManager.VT_TRUCK
                elif cllxfl in buss:
                    vehicle_type = VdJsonManager.VT_BUS
                arrs_c = xlwz.split(',')
                box = [int(arrs_c[0]), int(arrs_c[1]), int(arrs_c[2]), int(arrs_c[3])]
                if box[0] < 0:
                    box[0] = 0
                if box[1] < 0:
                    box[1] = 0
                try:
                    VdJsonManager.s_lock.acquire() # 获取锁以进行目录操作
                    VdJsonManager.s_num, dst_cut_fn = FileTreeFolderSaver.get_dst_fn('{0}/{1}/{2}'.format(cut_img_folder, head_tail, vehicle_type), full_fn, VdJsonManager.s_num)
                    croped_img = VdJsonManager.crop_and_resize_img(full_fn, box)
                    cv2.imwrite(dst_cut_fn, croped_img)
                    VdJsonManager.s_lock.release()
                except Exception as ex:
                    print('##### Exception {0};[{1}]'.format(ex, full_fn))
                    efd.write('{0}\n'.format(full_fn))
                    VdJsonManager.s_lock.release()
                if VdJsonManager.s_num % 100 == 0:
                    print('Thread_{0}: cut and save {1};'.format(idx, VdJsonManager.s_num))
            else:
                efd.write('{0}\n'.format(full_fn))

    @staticmethod
    def get_img_reid_feature_vector(full_fn):
        url = 'http://192.168.2.17:2222/vehicle/function/recognition'
        data = {'TPLX': 1, 'GCXH': 123131318}
        files = {'TPWJ': (full_fn, open(full_fn, 'rb'))}
        resp = requests.post(url, files=files, data = data)
        json_obj = json.loads(resp.text)
        return json_obj
            
    @staticmethod
    def parse_vd_json_data(data):
        cllxfls = ['11', '12', '13', '14', '21', '22']
        if ('VEH' not in data) or len(data['VEH']) < 1:
            return None, None, None
        else:
            # 找到面积最大的检测框作为最终检测结果
            max_idx = -1
            max_area = 0
            for idx, veh in enumerate(data['VEH']):
                cllxfl = veh['CXTZ']['CLLXFL'][:2]
                if cllxfl in cllxfls:
                    box_str = veh['WZTZ']['CLWZ']
                    arrs_a = box_str.split(',')
                    x1, y1, w, h = int(arrs_a[0]), int(arrs_a[1]), int(arrs_a[2]), int(arrs_a[3])
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_idx = idx
            if max_idx < 0:
                return None, None, None
            else:
                return data['VEH'][max_idx]['WZTZ']['PSFX'], \
                        data['VEH'][max_idx]['CXTZ']['CLLXFL'][:2],\
                        data['VEH'][max_idx]['WZTZ']['CLWZ']
        
    @staticmethod
    def run_vd_cut_save_on_es20200923():
        print('利用Python程序完成整个切图流程（处理20200923无锡所测试错误图片）')
        base_path = Path('./support/ds_files/errorPicPP')
        with open('./support/es20200923_images.txt', 'w+', encoding='utf-8') as wfd:
            for file_obj in base_path.iterdir():
                full_fn = str(file_obj)
                wfd.write('./{0}\n'.format(full_fn))
        i_debug = 1
        if 1 == i_debug:
            return
        # 统计已经完成切图的图片文件集合
        cutted_image_set = set()
        txts_num = 2
        VdJsonManager.s_num = 0
        VdJsonManager.s_lock = threading.RLock()
        miss_images_fd = open('./support/wxs_es20200923_miss_images.txt', 'w+', encoding='utf-8')
        #miss_images_fd = open('./support/raw_miss_images.txt', 'w+', encoding='utf-8')
        efd = open('./support/m900_error.txt', 'w+', encoding='utf-8')
        #efd = open('./support/raw_error.txt', 'w+', encoding='utf-8')
        # 将图片文件列表均匀分给20个文本文件
        fd = open('./support/es20200923_images.txt', 'r', encoding='utf-8')
        params = {
            'idx': 0, 'fd': fd, 'efd': efd, 
            'miss_images_fd': miss_images_fd,
            'cutted_image_set': cutted_image_set,
            'cut_img_folder': './support/ds_files/esc20200923'
        }
        thd = threading.Thread(target=VdJsonManager.vd_cut_save_thd, args=(params,))
        thd.start()
        thd.join()
        fd.close()
        miss_images_fd.close()
        efd.close()
        
        
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
