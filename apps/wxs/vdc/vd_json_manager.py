# 由于在车辆检测时，会将所有检测json保存在同一目录下，会造成
# 文件读写缓慢的问题，所以需要一个独立的线程，将Json文件定期
# 从该目录中拷贝出来，存储到一个按文件编号为层次结构的规整目录
# 下面。
from pathlib import Path
#

class VdJsonManager(object):
    RM_SAVE_JSONS_IN_TREE_FOLDER = 1
    
    def __init__(self):
        self.refl = 'apps.wxs.vdc.VdJsonManager'
        
    def start(self):
        mode = VdJsonManager.RM_SAVE_JSONS_IN_TREE_FOLDER
        if VdJsonManager.RM_SAVE_JSONS_IN_TREE_FOLDER == mode:
            VdJsonManager.save_jsons_in_tree_folder()
        else:
            print('unknow mode')
    
    @staticmethod
    def save_jsons_in_tree_folder():
        '''
        将车辆检测Json文件从一个目录拷贝到树形目录下
        '''
        # 将Json文件以规整的目录格式存放
        base_path = Path('/media/zjkj/work/fgvc_dataset/raw_json')
        dst_folder = '/media/zjkj/work/fgvc_dataset/vdc0907/json_500'
        file_id = 0
        for jf_obj in base_path.iterdir():
            full_fn = str(jf_obj)
            file_id = FileTreeFolderSaver.save(dst_folder, full_fn, file_id)
            if file_id % 100 == num:
                print('处理完成{0}个文件'.format(file_id))