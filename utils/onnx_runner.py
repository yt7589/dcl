import onnxruntime
import numpy as np
from PIL import Image
import glob
# define the mean and std of the images
mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
std = np.array([0.229, 0.224, 0.225]).reshape([1,1,3])

def load_img( img_file):
    with open(img_file, 'rb') as f:
        with Image.open(f) as img:
            img_obj = img.convert('RGB')
            img_obj = img_obj.resize((224, 224), Image.BILINEAR)
    raw = (np.asarray(img_obj) / 255.0 - mean)/std
    return np.transpose(raw,[2,0,1])

def display_onnx(sess):
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

sess = onnxruntime.InferenceSession('dcl_v008_1.onnx')
#img_file = '/media/zjkj/work/yantao/zjkj/test_ds/00/00/白#06_WJG00300_016_长城_M4_2012-2014_610500200969341894.jpg'
#img_file = '/media/zjkj/work/yantao/zjkj/test_ds/00/00/白#02_陕EMH808_005_宝马_5系_2014_610500200969347480.jpg'
num = 0
correct_num = 0
with open('./datasets/CUB_200_2011/anno/bid_brand_test_ds_082801.txt', 'r', encoding='utf-8') as tfd:
    for line in tfd:
        line = line.strip()
        arrs_a = line.split('*')
        full_fn = arrs_a[0]
        gt_sim_bmy_id = int(arrs_a[1])
        gt_brand_idx = int(arrs_a[2])
        img = load_img(full_fn)
        X = img.reshape((1, 3, 224, 224))
        X = X.astype(np.float32)
        result = sess.run(['brands', 'bmys'], {'data': X})
        net_brand_idx = np.argmax(result[0], axis=1)
        net_sim_bmy_id = np.argmax(result[1], axis=1)
        if gt_brand_idx == net_brand_idx:
            correct_num += 1
        num += 1
        print('brand: {0}; bmy: {1};'.format(net_brand_idx, net_sim_bmy_id))
print('brand accuracy: {0};'.format(correct_num / num))

'''  
for img_file in glob.glob("/media/zjkj/work/yantao/zjkj/test_ds/00/00/*.jpg"):
    img = load_img(img_file)
    #print('img: {0}; {1};'.format(type(img), img.shape))
    X = img.reshape((1, 3, 224, 224))
    X = X.astype(np.float32)
    result = sess.run(["brands", "bmys"], {"data": X})
    brand = np.argmax(result[0], axis=1)
    bmy = np.argmax(result[1], axis=1)
    print('brand: {0}; bmy: {1};'.format(brand, bmy))
'''

'''
result: (1, 169); (1, 2891);
brand: -0.050930239260196686;
bmy: 0.04837498068809509;
brand: [22]; bmy: [1926];

result: (1, 169); (1, 2891);
brand: -0.04043670743703842;
bmy: 0.03334234282374382;
brand: [51]; bmy: [1926];
'''