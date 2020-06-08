
import json
from pathlib import Path
from flask import request
from apps.admin.controller.flask_web import FlaskWeb
from utils.ds_manager import DsManager
from apps.admin.controller.c_brand import CBrand
from apps.admin.controller.c_model import CModel
from apps.admin.controller.c_bmy import CBmy
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_ggh_bmy import MGghBmy
from apps.admin.model.m_bmy import MBmy

class CGghBmy(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CGghBmy'

    @staticmethod
    def ggh_to_bmy_dict_api():
        rst = CGghBmy.ggh_to_bmy_dict()
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'ggh_num': rst['ggh_num'],
            'brand_num': rst['brand_num'],
            'model_num': rst['model_num'],
            'bmy_num': rst['bmy_num']
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def ggh_to_bmy_dict():
        brand_set = set()
        model_set = set()
        bmy_set = set()
        # 读出ggh_to_bmy_dict.txt内容
        ggh_to_bmy_dict = DsManager.get_ggh_to_bmy_dict()
        for k, v in ggh_to_bmy_dict.items():
            arrs0 = v.split('_')
            brand_name = arrs0[0]
            brand_id = CBrand.add_brand(brand_name)
            brand_set.add(brand_name)
            model_name = '{0}_{1}'.format(arrs0[0], arrs0[1])
            model_set.add(model_name)
            model_id = CModel.add_model(brand_id, model_name)
            bmy_set.add(v)
            bmy_id = CBmy.add_bmy(brand_id, model_id, v)
            print('process：{0}...'.format(v))
            ggh_bmy_vo = MGghBmy.get_ggh_bmy_by_code(k)
            if ggh_bmy_vo is None:
                ggh_bmy_id = MPkGenerator.get_pk('ggh_bmy')
                ggh_bmy_vo = {
                    'ggh_bmy_id': ggh_bmy_id,
                    'ggh_code': k,
                    'bmy_id': bmy_id
                }
                MGghBmy.insert(ggh_bmy_vo)
        rst = {
            'ggh_num': len(ggh_to_bmy_dict),
            'brand_num': len(brand_set),
            'model_num': len(model_set),
            'bmy_num': len(bmy_set)
        }
        return rst

    @staticmethod
    def process_error_ggh_bmys():
        ''' 
        处理冲突的公告号和品牌车型年款对应关系，由品牌_车型_年款
        从t_bmy表中求出bmy_id，然后以ggh_code为条件更新t_ggh_bmy
        中记录的bmy_id
        '''
        ggh_to_bmy_dict = DsManager.get_ggh_to_bmy_dict()
        with open('./logs/error_ggh1.txt', 'r', encoding='utf-8') as gb_fd:
            for line in gb_fd:
                arrs0 = line.split(':')
                ggh_code = arrs0[0]
                bmy_name = arrs0[1][:-1]
                ggh_to_bmy_dict[ggh_code] = bmy_name
        for k, v in ggh_to_bmy_dict.items():
            print('{0}:{1};'.format(k, v))
        with open('./work/ggh_to_bmy_dict.txt', 'w+', encoding='utf-8') as fd2:
            for k, v in ggh_to_bmy_dict.items():
                print('{0}:{1};'.format(k, v))
                fd2.write('{0}:{1}\n'.format(k, v))










    @staticmethod
    def process_bmy_dir(ggh_to_bmy_dict):
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        error_set = set()
        brand_set = set()
        model_set = set()
        bmy_set = set()
        error_dict = {}
        for brand_path in base_path.iterdir():
            brand_str = str(brand_path)
            arrs_b0 = brand_str.split('/')
            brand_name = arrs_b0[-1]
            for model_path in brand_path.iterdir():
                model_str = str(model_path)
                arrs_m0 = model_str.split('/')
                model_name = arrs_m0[-1]
                for year_path in model_path.iterdir():
                    year_str = str(year_path)
                    arrs_y0 = year_str.split('/')
                    year_name = arrs_y0[-1]
                    for file_obj in year_path.iterdir():
                        file_str = str(file_obj)
                        arrs0 = file_str.split('/')
                        arrs1 = arrs0[-1].split('_')
                        if arrs1[0].startswith('白') or arrs1[0].startswith('夜'):
                            continue
                        arrs2 = arrs1[0].split('#')
                        ggh = arrs2[0]
                        bmy = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                        brand_set.add(brand_name)
                        model_set.add('{0}_{1}'.format(brand_name, model_name))
                        bmy_set.add(bmy)
                        if ggh not in ggh_to_bmy_dict:
                            ggh_to_bmy_dict[ggh] = bmy
                        else:
                            bmy0 = ggh_to_bmy_dict[ggh]
                            if bmy != bmy0:
                                error_set.add(ggh)
                                error_dict[ggh] = 'org:{0}<=>{1}(imported vehicles);'.format(bmy0, bmy)
                        #print('ggh: {0};'.format(ggh))
        with open('./logs/error_ggh.txt', 'w+', encoding='utf-8') as e_fd:
            for k, v in error_dict.items():
                e_fd.write('{0}:{1}\n'.format(k, v))
        print('共有{0}条公告号记录，冲突记录{1}条！'.format(len(ggh_to_bmy_dict), len(error_set)))
        return ggh_to_bmy_dict
        