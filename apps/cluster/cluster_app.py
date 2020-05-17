# 图像聚类主程序
from apps.cluster.image_dbscan import ImageDbscan

class ClusterApp(object):
    def __init__(self):
        self.name = 'apps.cluster.ClusterApp'

    def startup(self):
        engine = ImageDbscan()
        engine.run()