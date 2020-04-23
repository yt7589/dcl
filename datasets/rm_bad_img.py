"""
清洗数据，删除不能打开的图片
author@liying
"""

import cv2 
import os 
from tqdm import tqdm 
import shutil
import argparse

#读取图片目录下 所有'.jpg'后缀的图片所在文件夹路径
def get_img_file(file_name):
    file_paths = []
    for root, dirs, files in os.walk(file_name): 
        for name in files: 
            if name.endswith('.jpg'): 
                file_paths.append(root)
    return list(set(file_paths))

#移除不能读取到图片内容的数据
def rm_bad_imgs(data_path):
    images = sorted([f for f in os.listdir(data_path) if
                   os.path.isfile(os.path.join(data_path, f)) and f.endswith('.jpg')])
    for i in range(len(images)):
        image_name = images[i] 
        img_data =  cv2.imread(os.path.join(data_path,image_name))
        if  img_data is None:     
            print('The image %s is damaged!!!'%image_name)
            if not os.path.exists('bad_img2'):
                os.mkdir('bad_img2')
            shutil.move(os.path.join(data_path,image_name), 'bad_img2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, required=True, help='the original data file name')
    args = parser.parse_args()
    file_name = args.file 
    data_paths = sorted(get_img_file(file_name))
    
    print("**********There is %d image files in the %s."%(len(data_paths), file_name))
    for i in tqdm(range(0, len(data_paths))):
        print("**********dealing with the file %d**********"%i)
        data_path = data_paths[i] + '/'
        print(data_path)
        rm_bad_imgs(data_path)            
        print('Finished %d !!!'%i)  
