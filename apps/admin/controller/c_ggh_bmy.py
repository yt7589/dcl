
import json
from pathlib import Path
from flask import request
from apps.admin.controller.flask_web import FlaskWeb
from utils.ds_manager import DsManager

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
        # 读出ggh_to_bmy_dict.txt内容
        ggh_to_bmy_dict = DsManager.get_ggh_to_bmy_dict()
        CGghBmy.process_bmy_dir(ggh_to_bmy_dict)
        '''
        brand_set = set()
        for k, v in ggh_to_bmy_dict.items():
            arrs0 = v.split('_')
            brand_name = arrs0[0]
            brand_set.add(brand_name)
        print('ggh_num={0}; brand_num={1};'.format(len(ggh_to_bmy_dict), len(brand_set)))
        '''
        rst = {
            'ggh_num': 201,
            'brand_num': 202,
            'model_num': 203,
            'bmy_num': 104
        }
        return rst
    @staticmethod
    def process_bmy_dir(ggh_to_bmy_dict):
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        error_set = set()
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
        