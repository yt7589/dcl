# 将图片文件保存到Lmdb中以提高训练效率。通过PIL读出图片
# 文件，将内容以合路径文件名为Key，内容为Value保存到数据
# 库中。
import lmdb
import matplotlib.pyplot as plt
import PIL.Image as Image

class ImageLmdb(object):
    s_env = None
    
    def __init__(self):
        self.refl = 'datasets.ImageLmdb'
        
    @staticmethod
    def initialize_lmdb():
        ImageLmdb.s_env = lmdb.open('./support/ds_image.db', map_size=2099511627776)
        
    @staticmethod
    def destroy():
        ImageLmdb.s_env.close()
        
    @staticmethod
    def demo():
        print('试验LMDB技术...')
        ImageLmdb.initialize_lmdb()
        print('evn={0};'.format(ImageLmdb.s_env))
        img_full_fn = '/media/zjkj/work/yantao/fgvc/dcl/support/datasets/train/head/bus/d00/d00/d00/d08/d76/JX6580TA-M5_贵AV760A_02_520000102291_520000104328453641.jpg'
        # 读取文件
        with open(img_full_fn, 'rb') as f:
            with Image.open(f) as img:
                img_obj = img.convert('RGB')
        # 将对象保存到lmdb中
        txn = ImageLmdb.s_env.begin(write=True)
        txn.put(key=img_full_fn, value=img_obj)
        txn.commit()
        # 从lmdb中读出图像对像
        img_obj_db = txn.get(key=img_full_fn)
        plt.subplot(1, 2, 1)
        plt.imshow(img_obj)
        plt.subplot(1, 2, 2)
        plt.imshow(img_obj_db)
        plt.show()
        ImageLmdb.destroy()
        
    