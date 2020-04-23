import os 
import argparse
import cv2  

def get_img_file(file_name):
    file_paths = []
    for root, dirs, files in os.walk(file_name): 
        for name in files: 
            if name.endswith('.jpg'): 
                file_paths.append(root)
    return list(set(file_paths))

def write_imgpath_to_txt(data_path, onehot_num, txt_file):
    images = sorted([f for f in os.listdir(data_path) if
                   os.path.isfile(os.path.join(data_path, f)) and f.endswith('.jpg')])
    for image in images:
        #if cv2.imread(os.path.join(data_path, image)) is not None:
        txt_file.write(os.path.join(data_path, image) + ' ' + str(onehot_num) +'\n')
            #print(os.path.join(data_path, image) + ' ' + str(onehot_num))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, required=True, help='the original data file name')
    args = parser.parse_args()
    file_name = args.file
    data_paths = sorted(get_img_file(file_name))
    f = open('img_labels.txt', 'w')
    print("**********There is %d image files in the %s."%(len(data_paths), file_name))
    for i in range(0, len(data_paths)):
        print("**********reading the file %d**********"%i)
        data_path = data_paths[i] + '/'
        print(data_path)
        write_imgpath_to_txt(data_path, i, f)          
        print('Finished %d !!!'%i)           
    f.close()
