# 
import os
import json
from pathlib import Path
import shutil

class FileUtil(object):
    def __init__(self):
        self.refl = 'apps.wxs.fu.FileUtil'
        
    @staticmethod
    def get_files_in_subfolders_dict(base_folder):
        print('获取文件名与全路径文件名对应关系的字典...')
        img_file_to_full_fn = {}
        num = 0
        for dirpath, dirnames, filenames in os.walk(base_folder):
            for fn in filenames:
                full_fn = '{0}/{1}'.format(dirpath, fn)
                img_file_to_full_fn[fn] = full_fn
                num += 1
                if num % 1000 == 0:
                    print('处理{0}个文件'.format(num))