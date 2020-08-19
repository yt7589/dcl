#
import os
import random
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt

class AtntFaceDs(Dataset):
    def __init__(self, imageFolderDataset,transform=None,should_invert=True):
        self.name = ''
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        '''
        # 人脸识别黑白图像
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        '''
        # 车辆识别彩色图像
        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)





    def convert_pgm_to_png(self):
        print('将pgm文件转换为png格式文件')
        file_sep = '\\'
        base_path = Path('e:/work/tcv/projects/dcl/datasets/siamese/faces_pgm')
        ds_folder = 'e:/work/tcv/projects/dcl/datasets/siamese/faces'
        for trt_obj in base_path.iterdir():
            for sub_obj in trt_obj.iterdir():
                if sub_obj.is_dir():
                    for pgm_obj in sub_obj.iterdir():
                        full_fn = str(pgm_obj)
                        if pgm_obj.is_file() and full_fn.endswith(('pgm')):
                            arrs_a = full_fn.split(file_sep)
                            pgm_file = arrs_a[-1]
                            sub_folder = arrs_a[-2]
                            ds_type = arrs_a[-3]
                            print('数据集类型：{0}; 子目录：{1}; 文件名：{2};'.format(ds_type, sub_folder, pgm_file))
                            dst_ds_folder = '{0}/{1}'.format(ds_folder, ds_type)
                            if not os.path.exists(dst_ds_folder):
                                os.mkdir(dst_ds_folder)
                            ds_sub_folder = '{0}/{1}'.format(dst_ds_folder, sub_folder)
                            if not os.path.exists(ds_sub_folder):
                                os.mkdir(ds_sub_folder)
                            img_obj = Image.open(full_fn)
                            img_obj.save('{0}/{1}.png'.format(ds_sub_folder, pgm_file[:-4]))