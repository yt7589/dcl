# 由于在车辆检测时，会将所有检测json保存在同一目录下，会造成
# 文件读写缓慢的问题，所以需要一个独立的线程，将Json文件定期
# 从该目录中拷贝出来，存储到一个按文件编号为层次结构的规整目录
# 下面。
import threading

class VdJsonSaver(object):
    def __init__(self):
        self.refl = 'apps.wxs.vdc.VdJsonSaver'
        
    def start(self):
        '''
        以启动规整化保存线程，按指定时间间隔运行，从检测结果目录
        拷贝出Json文件，放置到规整化目录下
        '''
        params1 = {
            'userId': 1008,
            'userName': '测试'
        }
        save_thd = threading.Thread(target=self.move_save_thd, args=(params1))
        save_thd.start()
        save_thd.join()
        
    def move_save_thd(self, args=()):
        print('参数：{0}-{1};'.format(args[0]['userId'], args[0]['userName']))