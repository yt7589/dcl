# onnx模型导出工具
import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import onnxruntime
import PIL.Image as Image
#from onnxruntime.datasets import get_example
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
        self.avgpool = nn.AvgPool2d(output_size=1) # nn.AdaptiveAvgPool2d(output_size=1)
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
        # 运行onnx
        self.run_onnx()

    def run_onnx(self):
        #example_model = onnxruntime.datasets.get_example('e:/temp/dcl_0822_n2.onnx')
        sess = onnxruntime.InferenceSession('dcl_v002_1.onnx')
        # input
        input_name = sess.get_inputs()[0].name
        print("Input name  :", input_name)
        input_shape = sess.get_inputs()[0].shape
        print("Input shape :", input_shape)
        input_type = sess.get_inputs()[0].type
        print("Input type  :", input_type)
        # output
        output_name0 = sess.get_outputs()[0].name
        print("Output0 name  :", output_name0)  
        output_shape0 = sess.get_outputs()[0].shape
        print("Output0 shape :", output_shape0)
        output_type0 = sess.get_outputs()[0].type
        print("Output0 type  :", output_type0)
        # output2
        output_name1 = sess.get_outputs()[1].name
        print("Output1 name  :", output_name1)  
        output_shape1 = sess.get_outputs()[1].shape
        print("Output1 shape :", output_shape1)
        output_type1 = sess.get_outputs()[1].type
        print("Output1 type  :", output_type1)
        # 
        #X = torch.rand(8, 3, 224, 224) #.cuda()
        img_file = '/media/zjkj/work/yantao/zjkj/test_ds/00/00/白#06_WJG00300_016_长城_M4_2012-2014_610500200969341894.jpg'
        img = self.load_img(img_file)
        print('img: {0};'.format(type(img)))
        X = img.reshape((1, 3, 224, 224))
        X = X.astype(np.float32)
        result = sess.run([output_name0, output_name1], {input_name: X})
        brand = np.argmax(result[0], axis=1)
        bmy = np.argmax(result[1], axis=1)
        print('result: {0}; {1};'.format(brand, bmy))

    def load_img(self, img_file):
        with open(img_file, 'rb') as f:
            with Image.open(f) as img:
                img_obj = img.convert('RGB')
                img_obj.resize((224, 224), Image.BILINEAR)
        return np.asarray(img_obj)