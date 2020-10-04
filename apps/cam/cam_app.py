#
import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import PIL.Image as Image
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from transforms import transforms
#
from apps.cam.core.feature_extractor import FeatureExtractor
from apps.cam.core.model_outputs import ModelOutputs
from apps.cam.core.grad_cam import GradCam

#
from apps.wxs.wxs_app import WxsApp

class CamApp(object):

    def __init__(self):
        self.refl = 'apps.cam.CamApp'

    def startup(self, args):
        i_debug = 18
        if 1 == i_debug:
            # 为无锡所招标预留功能开发
            app = WxsApp()
            app.startup(args)
            return
        print('模型热力图绘制应用 v0.0.6')
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        args = self.parse_args()
        # arg_dict = vars(args)
        args.train_num_workers = 0
        args.val_num_workers = 0
        print(args, flush=True)
        Config = LoadConfig(args, 'train')
        Config.cls_2 = args.cls_2
        Config.cls_2xmul = args.cls_mul
        assert Config.cls_2 ^ Config.cls_2xmul
        transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
        # inital dataloader
        train_set = dataset(Config = Config,\
                            anno = Config.train_anno,\
                            common_aug = transformers["common_aug"],\
                            swap = transformers["swap"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["train_totensor"],\
                            train = True)
        trainval_set = dataset(Config = Config,\
                            anno = Config.val_anno,\
                            common_aug = transformers["None"],\
                            swap = transformers["None"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["val_totensor"],\
                            train = False,
                            train_val = True)
        val_set = dataset(Config = Config,\
                          anno = Config.val_anno,\
                          common_aug = transformers["None"],\
                          swap = transformers["None"],\
                            swap_size=args.swap_num, \
                          totensor = transformers["test_totensor"],\
                          test=True)
        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                    batch_size=args.train_batch,\
                                                    shuffle=True,\
                                                    num_workers=args.train_num_workers,\
                                                    collate_fn=collate_fn4train if not Config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if Config.use_backbone else False,
                                                    pin_memory=True)
        setattr(dataloader['train'], 'total_item_len', len(train_set))
        dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if Config.use_backbone else False,
                                                    pin_memory=True)
        setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
        setattr(dataloader['trainval'], 'num_cls', Config.num_brands)
        dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4test if not Config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if Config.use_backbone else False,
                                                    pin_memory=True)
        setattr(dataloader['val'], 'total_item_len', len(val_set))
        setattr(dataloader['val'], 'num_cls', Config.num_brands)
        cudnn.benchmark = True
        print('Choose model and train set', flush=True)
        print('Choose model and train set', flush=True)
        model = MainModel(Config)

        # load model
        if (args.resume is None) and (not args.auto_resume):
            print('train from imagenet pretrained models ...', flush=True)
        else:
            if not args.resume is None:
                resume = args.resume
                print('load from pretrained checkpoint %s ...'% resume, flush=True)
            elif args.auto_resume:
                resume = self.auto_load_resume(Config.save_dir)
                print('load from %s ...'%resume, flush=True)
            else:
                raise Exception("no checkpoints to load")

            model_dict = model.state_dict()
            pretrained_dict = torch.load(resume)
            print('train.py Ln193 resume={0};'.format(resume))
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        print('Set cache dir', flush=True)
        time = datetime.datetime.now()
        filename = '%s_%d%d%d_%s'%(args.cam, time.month, time.day, time.hour, Config.dataset)
        save_dir = os.path.join(Config.save_dir, filename)
        print('save_dir: {0} + {1};'.format(Config.save_dir, filename))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.cuda()
        cam_model = model.model
        model = nn.DataParallel(model)
        # optimizer prepare
        if Config.use_backbone:
            ignored_params = list(map(id, model.module.classifier.parameters())) \
                        + list(map(id, model.module.brand_clfr.parameters()))
        else:
            ignored_params1 = list(map(id, model.module.classifier.parameters()))
            ignored_params1x = list(map(id, model.module.brand_clfr.parameters()))
            ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
            ignored_params3 = list(map(id, model.module.Convmask.parameters()))
            ignored_params = ignored_params1 + ignored_params1x + ignored_params2 + ignored_params3 
        print('the num of new layers:', len(ignored_params), flush=True)
        base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
        lr_ratio = args.cls_lr_ratio
        base_lr = args.base_lr
        momentum = 0.9
        if Config.use_backbone:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': base_lr},
                                   {'params': model.module.brand_clfr.parameters(), 'lr': base_lr}
                                   ], lr = base_lr, momentum=momentum)
        else:
            optimizer = optim.SGD([{'params': base_params},
                                   {'params': model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.brand_clfr.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.classifier_swap.parameters(), 'lr': lr_ratio*base_lr},
                                   {'params': model.module.Convmask.parameters(), 'lr': lr_ratio*base_lr},
                                  ], lr = base_lr, momentum=momentum)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)
        # *******************
        # *******************
        print('model: {0};'.format(cam_model))
        grad_cam = GradCam(model=cam_model, feature_module=cam_model[7], \
                       target_layer_names=["2"], use_cuda=True)
        # 读入图片数据
        img = None
        img_file = '/media/ps/0A9AD66165F33762/yantao/dcl/support/ds_files/wxs_ds/head/car/d00/d00/d00/d00/d96/SC7168CH5_冀B591C5_02_120000100604_120000702916290242.jpg'
        with open(img_file, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
                
        crop_reso = 224
        to_tensor = transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            # ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        input = to_tensor(img).reshape(1, 3, 224, 224)
        print('input: {0};'.format(input.shape))
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        mask = grad_cam(input, target_index)
        
        print('^_^ The End! 002 ^_^')
        
        
        
    def auto_load_resume(self, load_dir):
        folders = os.listdir(load_dir)
        #date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
        date_list = [int(x.split('_')[2]) for x in folders]
        choosed = folders[date_list.index(max(date_list))]
        weight_list = os.listdir(os.path.join(load_dir, choosed))
        acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
        acc_list = [float(x) for x in acc_list]
        choosed_w = weight_list[acc_list.index(max(acc_list))]
        return os.path.join(load_dir, choosed, choosed_w)
        
    # parameters setting
    def parse_args(self):
        parser = argparse.ArgumentParser(description='dcl parameters')
        parser.add_argument('--data', dest='dataset',
                            default='CUB', type=str)
        parser.add_argument('--epoch', dest='epoch',
                            default=360, type=int)
        parser.add_argument('--backbone', dest='backbone',
                            default='resnet50', type=str)
        parser.add_argument('--cp', dest='check_point',
                            default=1000, type=int)
        parser.add_argument('--sp', dest='save_point',
                            default=1000, type=int)
        parser.add_argument('--tb', dest='train_batch',
                            default=128, type=int)
        parser.add_argument('--tnw', dest='train_num_workers',
                            default=16, type=int)
        parser.add_argument('--vb', dest='val_batch',
                            default=256, type=int)
        parser.add_argument('--vnw', dest='val_num_workers',
                            default=16, type=int)
        parser.add_argument('--lr', dest='base_lr',
                            default=0.0008, type=float)
        parser.add_argument('--lr_step', dest='decay_step',
                            default=50, type=int)
        parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                            default=10.0, type=float)
        parser.add_argument('--start_epoch', dest='start_epoch',
                            default=0,  type=int)
        parser.add_argument('--detail', dest='cam',
                            default='cam', type=str)
        parser.add_argument('--size', dest='resize_resolution',
                            default=224, type=int)
        parser.add_argument('--crop', dest='crop_resolution',
                            default=224, type=int)
        parser.add_argument('--cls_2', dest='cls_2', default=True, 
                            action='store_true')
        parser.add_argument('--swap_num', default=[7, 7],
                        nargs=2, metavar=('swap1', 'swap2'),
                        type=int, help='specify a range')
        parser.add_argument('--auto_resume', dest='auto_resume',
                            default=False,
                            action='store_true')
                            
        parser.add_argument('--save', dest='resume',
                            default=None,
                            type=str)
        parser.add_argument('--cls_mul', dest='cls_mul',
                            action='store_true')
        args = parser.parse_args()
        return args