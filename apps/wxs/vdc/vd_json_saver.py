# 由于在车辆检测时，会将所有检测json保存在同一目录下，会造成
# 文件读写缓慢的问题，所以需要一个独立的线程，将Json文件定期
# 从该目录中拷贝出来，存储到一个按文件编号为层次结构的规整目录
# 下面。
import time
import threading
from pathlib import Path
#
from apps.wxs.fu.file_tree_folder.saver import FileTreeFolderSaver

class VdJsonSaver(object):
    def __init__(self):
        self.refl = 'apps.wxs.vdc.VdJsonSaver'
        
    def start(self):
        '''
        以启动规整化保存线程，按指定时间间隔运行，从检测结果目录
        拷贝出Json文件，放置到规整化目录下
        '''
        params = {
            'sleep_time': 1
        }
        save_thd = threading.Thread(target=VdJsonSaver.move_save_thd, args=(params,))
        save_thd.start()
        save_thd.join()
    
    @staticmethod
    def move_save_thd(params):
        nop_num = 0
        file_id = 0
        base_folder = '/media/zjkj/work/fgvc_dataset/vd_jsons'
        while True:
            base_path = Path('/media/zjkj/work/fgvc_dataset/raw_json')
            is_nop = True
            for jf_obj in base_path.iterdir():
                is_nop = False
                full_fn = str(if_obj)
                if jf_obj.is_file and full_fn.endswith(('json',)):
                    print('移动json文件:{0}...'.format(full_fn))
                    #shutil.move(full_fn, 
                    file_id = FileTreeFolderSaver.save_file(base_folder, file_id)
            if is_nop:
                nop_num += 1
            if nop_num > 5:
                break
            time.sleep(params['sleep_time'])