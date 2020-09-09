# 将图片文件保存到Lmdb中以提高训练效率。通过PIL读出图片
# 文件，将内容以合路径文件名为Key，内容为Value保存到数据
# 库中。
import pickle
from pathlib import Path
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
    def save_ds_imgs_to_lmdb():
        '''
        将数据集中图片全部保存到lmdb中
        '''
        base_path = Path('/media/zjkj/work/yantao/zjkj/test_ds_v1')
        txn = ImageLmdb.s_env.begin(write=True)
        for sf1 in base_path.iterdir():
            for sf2 in sf1.iterdir():
                for file_obj in sf2.iterdir():
                    full_fn = str(file_obj)
                    if file_obj.is_file() and full_fn.endswith(('jpg', 'jpeg')):
                        print('处理图片文件：{0};'.format(full_fn))
                        ImageLmdb.save_image_multi(txn, full_fn)
        txn.commit()
                        
    @staticmethod
    def save_image(img_full_fn):
        # 读取文件
        with open(img_full_fn, 'rb') as f:
            with Image.open(f) as img:
                img_obj = img.convert('RGB')
        txn = ImageLmdb.s_env.begin(write=True)
        txn.put(key=img_full_fn.encode(), value=pickle.dumps(img_obj))
        txn.commit()
                        
    @staticmethod
    def save_image_multi(txn, img_full_fn):
        # 读取文件
        with open(img_full_fn, 'rb') as f:
            with Image.open(f) as img:
                img_obj = img.convert('RGB')
        txn.put(key=img_full_fn.encode(), value=pickle.dumps(img_obj))
        
    @staticmethod
    def get_image(img_full_fn):
        txn = ImageLmdb.s_env.begin(write=False)
        return pickle.loads(txn.get(key=img_full_fn.encode()))
        
    @staticmethod
    def get_image_multi(txn, img_full_fn):
        #txn = ImageLmdb.s_env.begin(write=False)
        return pickle.loads(txn.get(key=img_full_fn.encode()))
        
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
        #txn = ImageLmdb.s_env.begin(write=True)
        #txn.put(key=img_full_fn.encode(), value=pickle.dumps(img_obj))
        #txn.commit()
        # 从lmdb中读出图像对像
        txn = ImageLmdb.s_env.begin(write=False)
        img_obj_db = pickle.loads(txn.get(key=img_full_fn.encode()))
        #plt.subplot(1, 2, 1)
        #plt.imshow(img_obj)
        plt.subplot(1, 2, 2)
        plt.imshow(img_obj_db)
        plt.show()
        ImageLmdb.destroy()
        
    