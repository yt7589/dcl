#.......................................................................................................................
#统计文件夹图片数量  写入txt
#.......................................................................................................................


import os

from progressbar import ProgressBar

path = '/home/zjkj/data/vehicle_type_v2d/vehicle_type_v2d'

original_images = []
# walk direction

for root, dirs, filenames in os.walk(path):
    print(root)

    for filename in filenames:
        original_images.append(os.path.join(root, filename))

original_images = sorted(original_images)

print('num:', len(original_images))

f = open('tongji_files_v2d.txt', 'w+')

error_images = []

progress = ProgressBar()

current_dirname = os.path.dirname(original_images[0])

file_num = 0

for filename in progress(original_images):

    dirname = os.path.dirname(filename)

    if dirname != current_dirname or filename == original_images[-1]:

        if filename == original_images[-1]:
            file_num += 1

        f.write('%s:\t%d\n' % (current_dirname, file_num))

        current_dirname = dirname

        file_num = 1

    else:
        file_num += 1
f.seek(0)

for s in f:
    print(s)

f.close()
