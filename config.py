import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

# pretrained model checkpoints
pretrained_model = {
    'resnet50' : './models/pretrained/resnet50-19c8e357.pth',
    'senet154' : './models/pretrained/senet154-c7b49a05.pth'
}

# transforms dict
def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################
        self.train_batch = args.train_batch
        self.val_batch = args.val_batch

        # put image data in $PATH/data
        # put annotation txt file in $PATH/anno

        if args.dataset == 'product':
            self.dataset = args.dataset
            self.rawdata_root = './../FGVC_product/data'
            self.anno_root = './../FGVC_product/anno'
            self.numcls = 2019
        elif args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = '/media/zjkj/work/vehicle_type_v2d/vehicle_type_v2d'
            self.anno_root = './datasets/CUB_200_2011/anno'
            #self.numcls = 0
            self.num_brands = 177 # 品牌数
            self.num_bmys = 3027 # 年款数
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/st_car/data'
            self.anno_root = './dataset/st_car/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './dataset/aircraft/data'
            self.anno_root = './dataset/aircraft/anno'
            self.numcls = 100
        else:
            raise Exception('dataset not defined ???')

        # annotation file organized as :
        # path/image_name cls_num\n
        # 正式环境
        train_ds_file = 'bid_brand_train_ds_082701.txt'
        val_ds_file = 'bid_brand_test_ds_082701.txt'
        test_ds_file = 'bid_brand_test_ds_082701.txt'
        '''
        # 精度测试
        val_ds_file = 'wxs_brands_ds.txt'
        test_ds_file = 'wxs_brands_ds.txt'
        '''

        if 'train' in get_list:
            self.train_anno = pd.read_csv(os.path.join(self.anno_root, train_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        if 'val' in get_list:
            '''
            # 所里品牌测试集
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'wxs_brands_cut_ds.txt'),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
            '''
            # 正式环境：品牌为主任务
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, val_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        if 'test' in get_list:
            '''
            # 所里品牌测试集
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'wxs_brands_cut_ds.txt'),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
            '''
            # 正式环境：品牌为主任务
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, test_ds_file),\
                                           sep="*",\
                                           header=None,\
                                           names=['ImageName', 'bmy_label', 'brand_label'])
        self.swap_num = args.swap_num

        self.save_dir = './net_model/'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.backbone = args.backbone

        self.use_dcl = True
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        # 当为True可以提升小样本类别精度
        self.use_focal_loss = False
        self.use_fpn = False
        self.use_hier = False

        self.weighted_sample = False
        self.cls_2 = False
        self.cls_2xmul = True

        self.task1_control_task2 = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)




