# 将图片文件保存到Lmdb中以提高训练效率。通过PIL读出图片
# 文件，将内容以合路径文件名为Key，内容为Value保存到数据
# 库中。
import lmdb

class ImageLmdb(object):
    s_env = None
    
    def __init__(self):
        self.refl = 'datasets.ImageLmdb'
        
    @staticmethod
    def initialize_lmdb():
        ImageLmdb.s_env = lmdb.open('./support/ds_image.db', , map_size=2099511627776)
        
    @staticmethod
    def demo():
        print('试验LMDB技术...')
        ImageLmdb.initialize_lmdb()
        print('evn={0};'.format(ImageLmdb.s_env))
        
    