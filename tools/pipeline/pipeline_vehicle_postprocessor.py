'''
Pipeline运行后会将每个图片的识别结果形成一个json文件，存放在指定目录下，
本方法以递归形式读出该目录下json文件，根据文件名，找到该图片文件对应的
正确分类，包括Top1（bmy_code）和品牌精度（brand_code），从json文件中
读出Top1和品牌结果，正确增加bmy_corrects和brand_corrects，最后通过计算
bmy_acc=bmy_corrects / total, brand_acc = brand_corrects / total
'''
import pymongo
import json
from pathlib import Path

def get_bmy_sim_org_dict():
    '''
    我们将3330个年款重新编为0~3329，称之为简化版bmy_id，我们同时维护一个
    字典，可以通过简化bmy_id查到其真实bmy_id
    '''
    bmy_sim_org_dict = {}
    with open('./config/bmy_sim_org_dict.txt', 'r', encoding='utf-8') as sofd:
        for line in sofd:
            line = line.strip()
            arrs0 = line.split(':')
            bmy_sim_org_dict[int(arrs0[0])] = int(arrs0[1])
    return bmy_sim_org_dict

def get_bmy_id_bmy_vo_dict():
    '''
    维护bmy_id和bmy_code、brand_code的对应关系
    '''
    bmy_id_bmy_vo_dict = {}
    with open('./config/bmys.txt', 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip()
            arrs0 = line.split('*')
            bmy_id = int(arrs0[0])
            bmy_id_bmy_vo_dict[bmy_id] = {
                'bmy_code': arrs0[1].strip(),
                'brand_code': arrs0[2].strip()
            }
    return bmy_id_bmy_vo_dict

def get_result_dict():
    result_dict = {}
    num = 0
    with open('./config/result_dict.txt', 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip()
            arrs0 = line.split('*')
            img_file = arrs0[0]
            result_dict[img_file] = {
                'bmy_code': arrs0[1],
                'brand_code': arrs0[2]
            }
            num += 1
    return result_dict, num

def calculate_result(result_dict, base_path, result):
    '''
    对指定目录下JSON文件，根据图片文件从result_dict中查出正确的年款编号和
    品牌编号，然后解析JSON文件，得到预测的年款编号和品牌编号，如果年款编号
    一致则年款正确数量加1，如果品牌编号一致则品牌正确数量加1
    '''
    for path_obj in base_path.iterdir():
        if path_obj.is_dir():
            calculate_result(result_dict, path_obj, result)
        else:
            full_fn = str(path_obj)
            arrs0 = full_fn.split('/')
            img_file_json = arrs0[-1]
            bmy_code, brand_code = parse_result_json(full_fn)
            img_file = img_file_json[:-5]
            if img_file in result_dict:
                bmy_vo = result_dict[img_file]
            else:
                bmy_vo = {
                    'bmy_code': 'x',
                    'brand_code': 'x'
                }
            gt_bmy_code = bmy_vo['bmy_code']
            gt_brand_code = bmy_vo['brand_code']
            if bmy_code == gt_bmy_code:
                result['bmy_corrects'] += 1
            if brand_code == gt_brand_code:
                result['brand_corrects'] += 1

def parse_result_json(json_file):
    '''
    从每张图片对应的JSON文件中解析并返回Top1（年款）编号和品牌编号
    '''
    bmy_code = ''
    brand_code = ''
    with open(json_file, 'r', encoding='utf-8') as jfd:
        json_str = jfd.read()
    json_obj = json.loads(json_str)
    bmy_code, brand_code = json_obj['VEH'][0]['CXTZ']['CXNK'], json_obj['VEH'][0]['CXTZ']['CLPP']
    return bmy_code, brand_code




'''
****************************************************************************************************************
****************************************************************************************************************
****************************  临时程序 *************************************************************************
****************************************************************************************************************
****************************************************************************************************************
'''

def generate_bmy_dict():
    '''
    将MongoDb中的内容从数据库中读出来保存为文本文件
    '''
    mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = mongo_client['stpdb']
    query_cond = {}
    fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                'brand_id': 1, 'brand_code': 1, 'model_id': 1, 
                'model_code': 1}
    recs = db['t_bmy'].find(query_cond, fields)
    with open('./config/bmys.txt', 'w+', encoding='utf-8') as fd:
        for rec in recs:
            fd.write('{0}*{1}*{2}\n'.format(rec['bmy_id'], rec['bmy_code'], rec['brand_code']))
    
def process_test_ds():
    bmy_sim_org_dict = get_bmy_sim_org_dict()
    bmy_id_bmy_vo_dict = get_bmy_id_bmy_vo_dict()
    with open('./config/result_dict.txt', 'w+', encoding='utf-8') as wfd:
        with open('./config/bid_brand_test_ds.txt', 'r', encoding='utf-8') as tfd:
            for line in tfd:
                line = line.strip()
                arrs0 = line.split('*')
                full_fn = arrs0[0]
                arrs1 = full_fn.split('/')
                img_file = arrs1[-1]
                sim_bmy_id = int(arrs0[1])
                bmy_id = bmy_sim_org_dict[sim_bmy_id] + 1
                bmy_vo = bmy_id_bmy_vo_dict[bmy_id]
                print('{0}*{1}*{2}'.format(img_file, bmy_vo['bmy_code'], bmy_vo['brand_code']))
                wfd.write('{0}*{1}*{2}\n'.format(img_file, bmy_vo['bmy_code'], bmy_vo['brand_code']))

def main(args):
    print('车辆识别精度计算程序 v0.0.1')
    result_dict, total = get_result_dict()
    base_path = Path('/media/zjkj/work/yantao/fgvc/dcl/logs/pipeline')
    result = {
        'bmy_corrects': 0,
        'brand_corrects': 0
    }
    calculate_result(result_dict, base_path, result)
    print('Top1精度：{0}({1})；品牌精度：{2}({3});'.format(
        result['bmy_corrects'] / total, result['bmy_corrects'],
        result['brand_corrects'] / total , result['brand_corrects']
    ))

if '__main__' == __name__:
    main({})
