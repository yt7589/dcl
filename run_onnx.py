import onnxruntime
import numpy as np
from PIL import Image
mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
std = np.array([0.229, 0.224, 0.225]).reshape([1,1,3])
def load_img( img_file):
    with open(img_file, 'rb') as f:
        with Image.open(f) as img:
            img_obj = img.convert('RGB')
            img_obj = img_obj.resize((224, 224), Image.BILINEAR)
    raw = (np.asarray(img_obj) / 255.0 - mean)/std
    return raw.reshape(3, 224, 224)
sess = onnxruntime.InferenceSession('model.onnx')
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
# output_name1 = sess.get_outputs()[1].name
# print("Output1 name  :", output_name1)
# output_shape1 = sess.get_outputs()[1].shape
# print("Output1 shape :", output_shape1)
# output_type1 = sess.get_outputs()[1].type
# print("Output1 type  :", output_type1)
#
# X = torch.rand(8, 3, 224, 224) #.cuda()
img_file = '/media/zjkj/work/yantao/zjkj/test_ds/00/00/白#06_WJG00300_016_长城_M4_2012-2014_610500200969341894.jpg'
#img_file = '/media/zjkj/work/yantao/zjkj/brand_clfrtest_ds/00/00/白#02_陕EMH808_005_宝马_5系_2014_610500200969347480.jpg'
img = load_img(img_file)
print('img: {0}; {1};'.format(type(img), img.shape))
X = img.reshape((1, 3, 224, 224))
X = X.astype(np.float32)
X =np.random.random([1, 3, 224, 224]).astype(np.float32)
result = sess.run([output_name0,"Add_170"], {input_name: X})
#print(result[0][0,0],result[1][0,0])
brand = np.argmax(result[0], axis=1)
#bmy = np.argmax(result[1], axis=1)
print('result: {0};'.format(brand))
print(result[1])
#print('result: {0}; {1};'.format(brand, bmy))


