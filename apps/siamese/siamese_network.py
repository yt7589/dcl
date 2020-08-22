#
import torch
from torch import nn
from torchvision import models
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
        self.model = getattr(models, self.backbone_arch)()
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