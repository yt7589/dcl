#
import argparse
from models.LoadModel import MainModel
from config import LoadConfig, load_data_transformers
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
        print('模型热力图绘制应用 v0.0.1')
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
        print('^_^ The End! ^_^')
        
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
                            default='', type=str)
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
    
        # 添加默认参数
        args.data = 'CUB'
        args.epoch = 360
        args.backbone = 'resnet50'
        args.cp = 1000
        args
        
        args.auto_resume = True
        args.backbone = 'resnet50'
        return args