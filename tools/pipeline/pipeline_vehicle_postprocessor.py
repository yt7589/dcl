'''
Pipeline运行后会将每个图片的识别结果形成一个json文件，存放在指定目录下，
本方法以递归形式读出该目录下json文件，根据文件名，找到该图片文件对应的
正确分类，包括Top1（bmy_code）和品牌精度（brand_code），从json文件中
读出Top1和品牌结果，正确增加bmy_corrects和brand_corrects，最后通过计算
bmy_acc=bmy_corrects / total, brand_acc = brand_corrects / total
'''

def process_test_ds(ds_file):
    with open('./config/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
        for line in sofd:
            line = line.strip()
    with open(ds_file, 'r', encoding='utf-8') as tfd:
        for line in tfd:
            line = line.strip()
            arrs0 = line.split('*')
            full_filename = arrs0[0]
            arrs1 = full_filename.split('/')
            img_file = arrs1[-1]
            sim_bmy_id = int(arrs0[1])
            bmy_id = 0
            bmy_vo = None
            bmy_code = bmy_vo['bmy_code']
            brand_code = bmy_vo['brand_code']

def main(args):
    print('main')
    process_test_ds('./config/bid_brand_test_ds.txt')

if '__main__' == __name__:
    main({})
