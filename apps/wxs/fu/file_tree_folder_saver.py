# 按文件编号将文件保存到每100个文件一个目录的树形目录结构中
from pathlib import Path

class FileTreeFolderSaver(object):
    def __init__(self):
        self.refl = 'apps.wxs.fu.FileTreeFolderSaver'
        
    @staticmethod
    def save_file(base_folder, src_full_fn, file_id):
        dst_folder = FileTreeFolderSaver.create_folder(base_folder, file_id)
        arrs_a = src_full.split('/')
        fn = arrs_a[-1]
        shutil.move(src_file, '{0}/{1}'.format(dst_folder, fn))
        return file_id+1
        
    @staticmethod
    def create_tree_folder(parent_folder, child_folder):
        dst_folder = '{0}/d{1}'.format(parent_folder, child_folder)
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)
        return dst_folder
        
    @staticmethod
    def create_folder(base_folder, file_id):
        full_str = '{0:012d}'.format(file_id)
        folder1 = FileTreeFolderSavercreate_tree_folder(base_folder, full_str[:2])
        folder2 = FileTreeFolderSavercreate_tree_folder(folder1, full_str[2:4])
        folder3 = FileTreeFolderSavercreate_tree_folder(folder2, full_str[4:6])
        folder4 = FileTreeFolderSavercreate_tree_folder(folder3, full_str[6:8])
        folder5 = FileTreeFolderSavercreate_tree_folder(folder4, full_str[8:10])
        return folder5