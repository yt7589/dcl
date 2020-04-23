#.......................................................................................................................
#删除图片数目少于100的图片
#.......................................................................................................................
import os

def del_img(path):
    for root, dirs, files in os.walk(path):
        print(root)

        for file in files:

            imgpath = os.path.join(root, file)
            list = os.listdir(root)

            if len(list) <= 50:
                os.remove(imgpath)

if __name__ == '__main__':
    path = r'/home/zjkj/data/vehicle_type_v2d/vehicle_type_v2d'
    del_img(path)

