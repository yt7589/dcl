#
import torch
from torch import nn
import pretrainedmodels
from apps.siamese.app_config import AppConfig

pretrained_model = {
    'resnet50' : './models/pretrained/resnet50-19c8e357.pth',
    'senet154' : './models/pretrained/senet154-c7b49a05.pth'
}

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone_arch = 'resnet50'
        self.fv_dim = 256
        self.fc_size = {'resnet50': 2048, 'resnet18': 512}
        self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
        self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(self.fc_size[self.backbone_arch], self.fv_dim, bias=False)

    def forward_once(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.classifier(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2










    def org_forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output



    

    def org_init(self):
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(AppConfig.img_channel, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            #nn.Linear(8*100*100, 500), # 人脸识别
            # 此处的8为最后层的通道数，我们做卷积操作均保持分辨率不变
            nn.Linear(8*AppConfig.img_w*AppConfig.img_h, 500), # 车辆识别
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))