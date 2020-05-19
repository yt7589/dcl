# 图像聚类主程序
import shutil
from apps.cluster.image_dbscan import ImageDbscan

class ClusterApp(object):
    def __init__(self):
        self.name = 'apps.cluster.ClusterApp'

    def startup(self):
        self.prepare_test_images()
        #engine = ImageDbscan()
        #engine.run()

    def prepare_test_images(self):
        '''
        将训练数据集文件中指定的图片，以a+类别编号形式命名并拷贝文件
        '''
        with open('./datasets/CUB_200_2011/anno/t1.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split('*')
                src_img = arrs0[0]
                fgvc_id = arrs0[-1][:-1]
                shutil.copy(src_img, './work/a{0}.jpg'.format(fgvc_id))