#.......................................................................................................................
#删除多级嵌套的文件夹
#.......................................................................................................................

import os
def delete_nested_folders(path):
    """    删除嵌套文件夹    :param path:    :return:    """
    if os.path.exists(path):
        x = os.listdir(path)  # 获得目录下的所有文件

        for i in x:
            endpath = os.path.join(path, i)
            try:
                if os.path.isdir(endpath):
                    delete_nested_folders(endpath)
                    print(endpath)
                    os.rmdir(endpath)
            except Exception as e:
                print('Exception', e)
if __name__ == '__main__':
    path = r'/home/zjkj/data/vehicle_type_v2d/'
    delete_nested_folders(path)
