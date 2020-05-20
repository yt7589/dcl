# 图像聚类主程序
import math
import shutil
import numpy as np
from apps.cluster.image_dbscan import ImageDbscan

class ClusterApp(object):
    def __init__(self):
        self.name = 'apps.cluster.ClusterApp'

    def startup(self):
        #self.prepare_test_images()
        #self.get_center_radius()
        engine = ImageDbscan()
        engine.run()

    def prepare_test_images(self):
        '''
        将训练数据集文件中指定的图片，以a+类别编号形式命名并拷贝文件
        '''
        idx = 1
        with open('./datasets/CUB_200_2011/anno/t1.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split('*')
                src_img = arrs0[0]
                fgvc_id = arrs0[-1][:-1]
                shutil.copy(src_img, './work/a{0}_{1}.jpg'.format(fgvc_id, idx))
                idx += 1

    def get_center_radius(self):
        features = np.loadtxt('./logs/cluster_features.txt', delimiter=' ')
        cf1 = features[:10, :]
        cf2 = features[10:20, :]
        cf3 = features[20:, :]
        print('cf1: {0}; cf2: {1}; cf3: {2};'.format(cf1.shape, cf2.shape, cf3.shape))
        center1 = np.mean(cf1, axis=0)
        d1_max = self.get_max_distance(center1, cf1)
        center2 = np.mean(cf2, axis=0)
        d2_max =self.get_max_distance(center2, cf2)
        center3 = np.mean(cf3, axis=0)
        d3_max = self.get_max_distance(center3, cf3)
        print('d1: {0}; d2: {1}; d3: {2};'.format(d1_max, d2_max, d3_max))

    def get_max_distance(self, center, points):
        d_max = -1
        for point in points:
            d = math.sqrt(np.sum(np.power(point - center, 2)))
            if d > d_max:
                d_max = d
        return d_max