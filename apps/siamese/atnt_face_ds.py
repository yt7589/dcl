#
import os
from PIL import Image
from pathlib import Path

class AtntFaceDs(object):
    def __init__(self):
        self.name = ''

    def convert_pgm_to_png(self):
        print('将pgm文件转换为png格式文件')
        file_sep = '\\'
        base_path = Path('e:/work/tcv/projects/dcl/datasets/siamese/faces_pgm')
        ds_folder = 'e:/work/tcv/projects/dcl/datasets/siamese/faces'
        for trt_obj in base_path.iterdir():
            for sub_obj in trt_obj.iterdir():
                if sub_obj.is_dir():
                    for pgm_obj in sub_obj.iterdir():
                        full_fn = str(pgm_obj)
                        if pgm_obj.is_file() and full_fn.endswith(('pgm')):
                            arrs_a = full_fn.split(file_sep)
                            pgm_file = arrs_a[-1]
                            sub_folder = arrs_a[-2]
                            ds_type = arrs_a[-3]
                            print('数据集类型：{0}; 子目录：{1}; 文件名：{2};'.format(ds_type, sub_folder, pgm_file))
                            dst_ds_folder = '{0}/{1}'.format(ds_folder, ds_type)
                            if not os.path.exists(dst_ds_folder):
                                os.mkdir(dst_ds_folder)
                            ds_sub_folder = '{0}/{1}'.format(dst_ds_folder, sub_folder)
                            if not os.path.exists(ds_sub_folder):
                                os.mkdir(ds_sub_folder)