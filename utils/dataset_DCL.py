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
    def __init__(self, Config, anno, swap_size=[7,7], common_aug=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        #self.numcls = Config.numcls
        self.num_task1 = Config.num_task1
        self.num_task2 = Config.num_task2
        self.dataset = Config.dataset
        self.use_cls_2 = Config.cls_2
        self.use_cls_mul = Config.cls_2xmul
        if isinstance(anno, pd.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.task1_labels = anno['task1_label'].tolist()
            self.task2_labels = anno['task2_label'].tolist()
            shuffle_samples(self.paths, self.task1_labels, self.task2_labels)
            self.task1_labels = list(map(int, self.task1_labels))
            self.task2_labels = list(map(int, self.task2_labels))
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.task1_labels = anno['task1_label']
            self.task2_labels = anno['task2_labels']
            shuffle_samples(self.paths, self.task1_labels, self.task2_labels)
            self.task1_labels = list(map(int, self.task1_labels))
            self.task2_labels = list(map(int, self.task2_labels))

        if train_val:
            #self.paths, self.labels = random_sample(self.paths, self.labels)
            self.paths = self.paths[:300]
            self.task1_labels = self.task1_labels[:300]
            self.task2_labels = self.task2_labels[:300]
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
            task1_label = self.task1_labels[item]
            task2_label = self.task2_labels[item]
            return img, task1_label, self.paths[item], task2_label
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
            task1_label = self.task1_labels[item]
            task2_label = self.task2_labels[item]
            if self.use_cls_mul:
                #label_swap = label + self.numcls
                #if not isinstance(label, int):
                #    print(type(label))
                task1_label_swap = int(task1_label) + self.numcls
            if self.use_cls_2:
                task1_label_swap = -1
            img_unswap = self.totensor(img_unswap)
            return img_unswap, img_swap, task1_label, task1_label_swap, swap_law1, swap_law2, self.paths[item], task2_label
        else:
            task1_label = self.task1_labels[item]
            task2_label = self.task2_labels[item]
            swap_law2 = [(i-(swap_range//2))/swap_range for i in range(swap_range)]
            task1_label_swap = task1_label
            img_unswap = self.totensor(img_unswap)
            return img_unswap, task1_label, task1_label_swap, swap_law1, swap_law2, self.paths[item], task2_label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

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
    task1_label = []
    task1_label_swap = []
    law_swap = []
    img_name = []
    task2_label = []
    #img_unswap, img_swap, label, label_swap, swap_law1, swap_law2, self.paths[item]
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        task1_label.append(sample[2])
        task1_label.append(sample[2])
        if sample[3] == -1:
            task1_label_swap.append(1)
            task1_label_swap.append(0)
        else:
            task1_label_swap.append(sample[2])
            task1_label_swap.append(sample[3])
        law_swap.append(sample[4])
        law_swap.append(sample[5])
        img_name.append(sample[-2])
        task2_label.append(sample[-1])
        task2_label.append(sample[-1])
    return torch.stack(imgs, 0), task1_label, task1_label_swap, law_swap, img_name, task2_label

def collate_fn4val(batch):
    imgs = []
    task1_label = []
    task1_label_swap = []
    law_swap = []
    img_name = []
    task2_label = []
    for sample in batch:
        imgs.append(sample[0])
        task1_label.append(sample[1])
        if sample[3] == -1:
            task1_label_swap.append(1)
        else:
            task1_label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-2])
        task2_label.append(sample[-1])
    return torch.stack(imgs, 0), task1_label, task1_label_swap, law_swap, img_name, task2_label

def collate_fn4backbone(batch):
    imgs = []
    task1_label = []
    img_name = []
    task2_label = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 7:
            task1_label.append(sample[2])
        else:
            task1_label.append(sample[1])
        img_name.append(sample[-2])
        task2_label.append(sample[-1])
    return torch.stack(imgs, 0), task1_label, img_name, task2_label


def collate_fn4test(batch):
    imgs = []
    task1_label = []
    img_name = []
    task2_label = []
    for sample in batch:
        imgs.append(sample[0])
        task1_label.append(sample[1])
        img_name.append(sample[-2])
        task2_label.append(sample[-1])
    return torch.stack(imgs, 0), task1_label, img_name, task2_label

def preprocess_anno():
    '''
    将训练样本集中的数据，从文件在硬盘中的顺序变为随机顺序，
    保证训练过程中的随机性
    '''
    anno_file = './datasets/CUB_200_2011/anno/ct_train_newd.txt'
    print(anno_file)
    anno = pd.read_csv(anno_file, sep="*", header=None, 
                names=['ImageName', 'task1_label', 'task2_label'])
    if isinstance(anno, pd.core.frame.DataFrame):
        paths = anno['ImageName'].tolist()
        task1_labels = anno['task1_label'].tolist()
        task2_labels = anno['task2_label'].tolist()
    elif isinstance(anno, dict):
        paths = anno['img_name']
        task1_labels = anno['task1_label']
        task2_labels = anno['task2_label'].tolist()
    shuffle_samples(paths, task1_labels, task2_labels)
    paths_len = len(paths)
    for i in range(0, paths_len):
        print('{0} => {1} {2};'.format(paths[i], task1_labels[i], task2_labels[i]))

def shuffle_samples(paths, task1_labels, task2_labels):
    paths_len = len(paths)
    task1_labels_len = len(task1_labels)
    print('{0}={1}?'.format(paths_len, task1_labels_len))
    for i in range(1, paths_len):
        rn = random.randint(0, paths_len-1)
        # 交换图片文件列表
        temp_path = paths[rn]
        paths[rn] = paths[i]
        paths[i] = temp_path
        # 交换年款列表
        temp_task2_label = task2_labels[rn]
        task2_labels[rn] = task2_labels[i]
        task2_labels[i] = temp_task2_label
        # 交换品牌列表
        temp_task1_label = task1_labels[rn]
        task1_labels[rn] = task1_labels[i]
        task1_labels[i] = temp_task1_label