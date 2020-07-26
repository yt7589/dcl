'''
Pipeline运行后会将每个图片的识别结果形成一个json文件，存放在指定目录下，
本方法以递归形式读出该目录下json文件，根据文件名，找到该图片文件对应的
正确分类，包括Top1（bmy_code）和品牌精度（brand_code），从json文件中
读出Top1和品牌结果，正确增加bmy_corrects和brand_corrects，最后通过计算
bmy_acc=bmy_corrects / total, brand_acc = brand_corrects / total
'''

def main(args):
    print('main')

if '__main__' == __name__:
    main({})
