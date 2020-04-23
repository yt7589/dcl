import os

def list_dir(fw,dir_path):

    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件

    for file in dir_files:

        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径

        if os.path.isfile(file_path):  # 如果是文件，就打印这个文件路径

            if file_path.endswith(".jpg"):

                fw.write(file_path+"\n")

        if os.path.isdir(file_path):  # 如果目录，就递归子目录

            list_dir(fw,file_path)

    return fw





if __name__ == '__main__':

    fw=open("images.txt","w")

    thesaurus_path = r"/home/zjkj/data/vehicle_type_v2d/vehicle_type_v2d"

    text_list = list_dir(fw, thesaurus_path)
