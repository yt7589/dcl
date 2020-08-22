# onnx模型导出工具
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict

class MainModel(nn.Module):
    '''
    定义需要导出的模型
    '''
    def __init__(self):
        super(MainModel, self).__init__()
        self.num_brands = 169 # 9000
        self.num_bmys = 2891
        self.backbone_arch = 'resnet50'
        
        self.model = getattr(models, self.backbone_arch)()
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_brands, bias=False)
        self.brand_clfr = nn.Linear(2048, self.num_bmys, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        brand_out = self.classifier(x)
        bmy_out = self.brand_clfr(x)
        return F.softmax(brand_out, dim=1), F.softmax(bmy_out, dim=1)

def load_model_wholepth_special(pth):
    # 直接加载
    net = MainModel()
    weight_value = torch.load(pth, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in weight_value.items():
        name = k[7:]  # remove `module.`
        if(name not in ["classifier_swap.weight", "Convmask.weight", "Convmask.bias"]):
            new_state_dict[name] = v

    # print(net)
    net.load_state_dict(new_state_dict, strict=True)
    return net

class OnnxExporter(object):
    def __init__(self):
        self.refl = '*'

    def export_onnx(self, model):
        # 载入模型
        print('OnnxExporter.export_onnx 1')
        pth_file = './net_model/training_descibe_82121_CUB/weights_0_43999_0.9744_0.9899.pth'
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('OnnxExporter.export_onnx 2')
        net = load_model_wholepth_special(pth_file)
        print('OnnxExporter.export_onnx 3')
        net.eval().float().to(dev)
        print('OnnxExporter.export_onnx 4')
        # 输出onnx
        example = torch.rand(8, 3, 224, 224).cuda()
        print('OnnxExporter.export_onnx 5')
        print(example.shape)
        net.train(False)
        net.eval()
        print('OnnxExporter.export_onnx 6')
        '''
        # 由example这个输入来决定batch
        torch.onnx.export(onnx_model, example, "dcl_0810_8.onnx", verbose=False,
                            input_names=["data"], output_names=["brands", "bmys"], \
                            training=False, opset_version=9,
                            do_constant_folding=True)
        '''
        # 动态batch
        torch.onnx.export(net, example, "dcl_0822_n2.onnx", verbose=False,
                            input_names=["data"], output_names=["brands", "bmys"], \
                            training=False, opset_version=9,
                            do_constant_folding=True,
                            dynamic_axes={"data":{0:"batch_size"},     # 批处理变量
                                    "brands":{0:"batch_size"},
                                    "bmys":{0:"batch_size"}})
        print('OnnxExporter.export_onnx 7')
        print('保存成功')