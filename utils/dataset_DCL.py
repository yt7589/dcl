# coding=utf8
from __future__ import division
import sys
import os
import torch
import torch.utils.data as data
import pandas as pd
import random
import PIL.Image as Image
from PIL import ImageStat

import pdb
# LMSV存储图片文件
from datasets.image_lmdb import ImageLmdb

def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list



class dataset(data.Dataset):
    # 数据集图片文件保存方法
    IMGM_FILE = 1
    IMGM_LMDB = 2
    
    def __init__(self, Config, anno, swap_size=[7,7], common_aug=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.img_mode = dataset.IMGM_LMDB
        if dataset.IMGM_LMDB == self.img_mode:
            print('使用lmdb保存图片...')
            ImageLmdb.initialize_lmdb() # 初始化LMDB
        self.root_path = Config.rawdata_root
        self.numcls = Config.num_brands
        self.num_brands = Config.num_brands
        self.num_bmys = Config.num_bmys
        self.dataset = Config.dataset
        self.use_cls_2 = Config.cls_2
        self.use_cls_mul = Config.cls_2xmul
        if isinstance(anno, pd.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.brand_labels = anno['brand_label'].tolist()
            self.bmy_labels = anno['bmy_label'].tolist()
            shuffle_samples(self.paths, self.brand_labels, self.bmy_labels)
            self.brand_labels = list(map(int, self.brand_labels))
            self.bmy_labels = list(map(int, self.bmy_labels))
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.brand_labels = anno['brand_label']
            self.bmy_labels = anno['bmy_labels']
            shuffle_samples(self.paths, self.brand_labels, self.bmy_labels)
            self.brand_labels = list(map(int, self.brand_labels))
            self.bmy_labels = list(map(int, self.bmy_labels))

        if train_val:
            #self.paths, self.labels = random_sample(self.paths, self.labels)
            self.paths = self.paths[:300]
            self.brand_labels = self.brand_labels[:300]
            self.bmy_labels = self.bmy_labels[:300]
        self.common_aug = common_aug
        self.swap = swap
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        #img_path = os.path.join(self.root_path, self.paths[item])
        img_path = self.paths[item]
        #print('pil_lod: {0};'.format(img_path))
        try:
            img = self.pil_loader(img_path)
        except OSError as ex:
            print('{0}: {1};'.format(img_path, ex))
            sys.exit(0)
        if self.test:
            img = self.totensor(img)
            brand_label = self.brand_labels[item]
            bmy_label = self.bmy_labels[item]
            return img, brand_label, self.paths[item], bmy_label
        img_unswap = self.common_aug(img) if not self.common_aug is None else img
        image_unswap_list = self.crop_image(img_unswap, self.swap_size)

        swap_range = self.swap_size[0] * self.swap_size[1]
        swap_law1 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]

        if self.train:
            img_swap = self.swap(img_unswap)
            image_swap_list = self.crop_image(img_swap, self.swap_size)
            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
            swap_law2 = []
            for swap_im in swap_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                swap_law2.append((index-(swap_range//2))/swap_range)
            img_swap = self.totensor(img_swap)
            brand_label = self.brand_labels[item]
            bmy_label = self.bmy_labels[item]
            if self.use_cls_mul:
                #label_swap = label + self.numcls
                #if not isinstance(label, int):
                #    print(type(label))
                brand_label_swap = int(brand_label) + self.numcls
            if self.use_cls_2:
                brand_label_swap = -1
            img_unswap = self.totensor(img_unswap)
            return img_unswap, img_swap, brand_label, brand_label_swap, swap_law1, swap_law2, self.paths[item], bmy_label
        else:
            brand_label = self.brand_labels[item]
            bmy_label = self.bmy_labels[item]
            swap_law2 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]
            brand_label_swap = brand_label
            img_unswap = self.totensor(img_unswap)
            return img_unswap, brand_label, brand_label_swap, swap_law1, swap_law2, self.paths[item], bmy_label

    def pil_loader(self,imgpath):
        if dataset.IMGM_FILE == self.img_mode:
            with open(imgpath, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')
        else:
            return ImageLmdb.get_image_multi(imgpath)

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)


def collate_fn4train(batch):
    imgs = []
    brand_label = []
    brand_label_swap = []
    law_swap = []
    img_name = []
    bmy_label = []
    #img_unswap, img_swap, label, label_swap, swap_law1, swap_law2, self.paths[item]
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        brand_label.append(sample[2])
        brand_label.append(sample[2])
        if sample[3] == -1:
            brand_label_swap.append(1)
            brand_label_swap.append(0)
        else:
            brand_label_swap.append(sample[2])
            brand_label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-2])
        bmy_label.append(sample[-1])
        bmy_label.append(sample[-1])
    return torch.stack(imgs, 0), brand_label, brand_label_swap, law_swap, img_name, bmy_label

def collate_fn4val(batch):
    imgs = []
    brand_label = []
    brand_label_swap = []
    law_swap = []
    img_name = []
    bmy_label = []
    for sample in batch:
        imgs.append(sample[0])
        brand_label.append(sample[1])
        if sample[3] == -1:
            brand_label_swap.append(1)
        else:
            brand_label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-2])
        bmy_label.append(sample[-1])
    return torch.stack(imgs, 0), brand_label, brand_label_swap, law_swap, img_name, bmy_label

def collate_fn4backbone(batch):
    imgs = []
    brand_label = []
    img_name = []
    bmy_label = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 7:
            brand_label.append(sample[2])
        else:
            brand_label.append(sample[1])
        img_name.append(sample[-2])
        bmy_label.append(sample[-1])
    return torch.stack(imgs, 0), brand_label, img_name, bmy_label


def collate_fn4test(batch):
    imgs = []
    brand_label = []
    img_name = []
    bmy_label = []
    for sample in batch:
        imgs.append(sample[0])
        brand_label.append(sample[1])
        img_name.append(sample[-2])
        bmy_label.append(sample[-1])
    return torch.stack(imgs, 0), brand_label, img_name, bmy_label

def preprocess_anno():
    '''
    将训练样本集中的数据，从文件在硬盘中的顺序变为随机顺序，
    保证训练过程中的随机性
    '''
    anno_file = './datasets/CUB_200_2011/anno/ct_train_newd.txt'
    print(anno_file)
    anno = pd.read_csv(anno_file, sep="*", header=None, 
                names=['ImageName', 'bmy_label', 'brand_label'])
    if isinstance(anno, pd.core.frame.DataFrame):
        paths = anno['ImageName'].tolist()
        brand_labels = anno['brand_label'].tolist()
        bmy_labels = anno['bmy_label'].tolist()
        #self.labels = list(map(int, self.labels))
    elif isinstance(anno, dict):
        paths = anno['img_name']
        brand_labels = anno['brand_label']
        bmy_labels = anno['bmy_label'].tolist()
    shuffle_samples(paths, brand_labels, bmy_labels)
    paths_len = len(paths)
    for i in range(0, paths_len):
        print('{0} => {1} {2};'.format(paths[i], brand_labels[i], bmy_labels[i]))

def shuffle_samples(paths, brand_labels, bmy_labels):
    paths_len = len(paths)
    brand_labels_len = len(brand_labels)
    print('{0}={1}?'.format(paths_len, brand_labels_len))
    for i in range(1, paths_len):
        rn = random.randint(0, paths_len-1)
        # 交换图片文件列表
        temp_path = paths[rn]
        paths[rn] = paths[i]
        paths[i] = temp_path
        # 交换年款列表
        temp_bmy_label = bmy_labels[rn]
        bmy_labels[rn] = bmy_labels[i]
        bmy_labels[i] = temp_bmy_label
        # 交换品牌列表
        temp_brand_label = brand_labels[rn]
        brand_labels[rn] = brand_labels[i]
        brand_labels[i] = temp_brand_label

