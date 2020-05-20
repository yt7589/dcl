# 采用DBScan算法进行图像自动聚类
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from transforms import transforms

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import config as dcl_config

class ImageDbscan(object):
    def __init__(self):
        self.name = 'apps.cluster.ImageDbscan'

    def run(self):
        print('DBScan图像自动聚类算法')
        #self.official_demo()
        self.cluster_images()

    def cluster_images(self):
        X = np.loadtxt('./logs/cluster_features.txt', delimiter=' ')
        labels_true = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ])
        '''
        X = np.array([
            [0.1, 0.15],
            [0.01, 0.05],
            [0.1, 0.1],
            [-0.1, -0.08],
            [-0.01, -0.002],
            #
            [2.01, 2.02],
            [1.99, 1.98],
            [2.1, 2.12],
            [2.08, 2.05],
            [1.98, 1.97],
            #
            [5.01, 5.02],
            [5.1, 5.09],
            [4.98, 4.99],
            [4.99, 5.0],
            [5.0, 4.97]
        ])
        labels_true = np.array([
            0,0,0,0,0,
            1,1,1,1,1,
            2,2,2,2,2
        ])
        '''
        tsne = TSNE(n_components=2, init='pca', perplexity=1.5)
        Y = tsne.fit_transform(X)
        print('Y: {0}; {1}; {2};'.format(type(Y), Y.shape, Y))
        plt.plot(Y[:, 0], Y[:, 1], 'o')
        plt.show()
        n_clusters_, n_noise_, labels, core_samples_mask = self.run_dbscan(X, eps=52.5, min_samples=5)
        print('labels: {0}; {1}; {2};'.format(type(labels), labels.shape, labels))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        '''
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
        '''

    def cluster_images0(self):
        # 循环读出所有图片文件名
        X = None
        labels_true = np.array([])
        path_obj = Path('e:/work/tcv/projects/tcv/data/raw_images')
        for file_obj in path_obj.iterdir():
            img_path = str(file_obj)
            cls_name = img_path.split('/')[-1][:2]
            if 'a1' == cls_name:
                labels_true = np.append(labels_true, 0)
            elif 'a2' == cls_name:
                labels_true = np.append(labels_true, 1)
            else:
                labels_true = np.append(labels_true, 2)
            x = self.load_image(img_path).numpy()
            if X is None:
                X = np.array([x])
            else:
                X = np.append(X, [x], axis=0)
        print('X: {0};'.format(X.shape))
        tsne = TSNE(n_components=2, init='pca', perplexity=30)
        Y = tsne.fit_transform(X)
        print('Y: {0}; {1}; {2};'.format(type(Y), Y.shape, Y))
        plt.plot(Y[:, 0], Y[:, 1], 'o')
        plt.show()

        '''
        n_clusters_, n_noise_, labels, core_samples_mask = self.run_dbscan(Y, eps=0.3*100, min_samples=2)
        print('labels_true: {0}; {1};'.format(type(labels_true), labels_true.shape))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(Y, labels))
        '''
        # 载入所有图片文件
        # 调用DBScan算法
        # 显示自动聚类结果
        '''
        img_path = 'E:/work/tcv/projects/datasets/StandCars/train/2/000090.jpg'
        x = self.load_image(img_path)
        print('x: {0};'.format(x.shape))
        '''

    def load_image(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img_obj:
                img = img_obj.convert('RGB')
        resize_resolution = 224
        crop_resolution = 224
        swap_num = [3, 3]
        transformers = dcl_config.load_data_transformers(resize_resolution, crop_resolution, swap_num)
        totensor = transformers['train_totensor']
        common_aug = transformers["common_aug"]
        img = common_aug(img)
        img_tensor = totensor(img)
        return img_tensor.view((img_tensor.shape[0]*img_tensor.shape[1]*img_tensor.shape[2],))

    def official_demo(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
        print('labels_true: {0}; {1};'.format(type(labels_true), labels_true.shape))
        n_clusters_, n_noise_, labels, core_samples_mask = self.run_dbscan(X)
        print('labels: {0}; {1}; {2};'.format(type(labels), labels.shape, labels))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
        self.draw_image(X, n_clusters_, labels, core_samples_mask)

    def run_dbscan(self, X, eps=0.3, min_samples=10):
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        return n_clusters_, n_noise_, labels, core_samples_mask
        

    def draw_image(self, X, n_clusters_, labels, core_samples_mask):
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()