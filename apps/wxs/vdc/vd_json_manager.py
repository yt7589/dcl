# 由于在车辆检测时，会将所有检测json保存在同一目录下，会造成
# 文件读写缓慢的问题，所以需要一个独立的线程，将Json文件定期
# 从该目录中拷贝出来，存储到一个按文件编号为层次结构的规整目录
# 下面。
import os
import json
from pathlib import Path
#
from apps.wxs.fu.file_tree_folder_saver import FileTreeFolderSaver

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
    
    @staticmethod
    def parse_vd_jsons():
        VdJsonManager.save_vd_json_fns()
        i_debug = 1
        if 1 == i_debug:
            return
        base_path = Path('/media/zjkj/work/fgvc_dataset/vdc0907/json_500')
        img_file_to_full_fn = VdJsonManager.get_raw_img_file_to_full_fn()
        num = 0
        cars = ['13', '14']
        trucks = ['21', '22']
        buss = ['11', '12']
        for sf1 in base_path.iterdir():
            for sf2 in sf1.iterdir():
                for sf3 in sf2.iterdir():
                    for sf4 in sf3.iterdir():
                        for sf5 in sf4.iterdir():
                            for jf_obj in sf5.iterdir():
                                full_fn = str(jf_obj)
                                psfx, cllxfl, xlwz = VdJsonManager.parse_vd_json(full_fn)
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
                                full_fn = img_file_to_full_fn[img_file]
                                print('img_file={0}: {1}; {2}; {3};'.format(full_fn, head_tail, vehicle_type, xlwz))
                                num += 1
                                if num > 3:
                                    return
        print('文件数量：{0};'.format(num))
        
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
                                    idx = num // 6000000
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
        if len(data['VEH']) < 1:
            return None
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
                return None
            else:
                return data['VEH'][max_idx]['WZTZ']['PSFX'], \
                        data['VEH'][max_idx]['CXTZ']['CLLXFL'][:2],\
                        data['VEH'][max_idx]['WZTZ']['CLWZ']
                                
                                
    @staticmethod
    def get_raw_img_file_to_full_fn():
        img_file_to_full_fn = {}
        iffn_file = './support/raw_iffn.txt'
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw')
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
            with open(iffn_file, 'w+', encoding='utf-8') as wfd:
                for k, v in img_file_to_full_fn.items():
                    wfd.write('{0}:{1}\n'.format(k, v))
        return img_file_to_full_fn
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
