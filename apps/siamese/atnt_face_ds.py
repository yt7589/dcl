#
from PIL import Image
from pathlib import Path

class AtntFaceDs(object):
    def __init__(self):
        self.name = ''

    def convert_pgm_to_png(self):
        print('将pgm文件转换为png格式文件')
        base_path = Path('e:/work/tcv/projects/dcl/datasets/siamese/faces_pgm')
        for trt_obj in base_path.iterdir():
            for sub_obj in trt_obj.iterdir():
                if sub_obj.is_dir():
                    for pgm_obj in sub_obj.iterdir():
                        full_fn = str(pgm_obj)
                        if pgm_obj.is_file() and full_fn.endswith(('pgm')):
                            print(full_fn)