import sys
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels

from config import pretrained_model

import pdb
#torch.backends.cudnn.benchmark = False

class MainModel(nn.Module):
    RUN_MODE_NORMAL = 100
    RUN_MODE_FEATURE_EXTRACT = 101

    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.num_brands = config.num_brands
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        self.run_mode = MainModel.RUN_MODE_NORMAL # 1-正常运行；2-输出最后一层的特征；
        print(self.backbone_arch)
        self.fc_size = {'resnet50': 2048, 'resnet18': 512}

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                # export TORCH_HOME="/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/models/pretrained/"
                # export TORCH_MODEL_ZOO="/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/models/pretrained/"
                #self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained='imagenet')
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=478)

        if self.backbone_arch == 'resnet18':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # 年款分类器
        self.classifier = nn.Linear(self.fc_size[self.backbone_arch], self.num_classes, bias=False)
        # 品牌分类器
        self.brand_clfr = nn.Linear(self.fc_size[self.backbone_arch], self.num_brands, bias=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(self.fc_size[self.backbone_arch], 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(self.fc_size[self.backbone_arch], 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(self.fc_size[self.backbone_arch], 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(self.fc_size[self.backbone_arch], self.num_classes, bias=False)

    def forward(self, x, last_cont=None, run_mode=RUN_MODE_NORMAL):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)
        x = self.avgpool(x)
        if MainModel.RUN_MODE_FEATURE_EXTRACT == run_mode:
            return x
        #x = x.view(x.size(0), -1)
        #x = x.view(x.size(0), x.size(1))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        out = []
        out.append(self.classifier(x))
        y_brand = self.brand_clfr(x)

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))
        out.append(y_brand)
        return out

