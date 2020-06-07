
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
        CGghBmy.process_bmy_dir()
        '''
        ggh_to_bmy_dict = DsManager.get_ggh_to_bmy_dict()
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
    def process_bmy_dir():
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        for brand_path in base_path.iterdir():
            for model_path in brand_path.iterdir():
                for year_path in model_path.iterdir():
                    for file_obj in year_path.iterdir():
                        file_str = str(file_obj)
                        arrs0 = file_str.split('/')
                        arrs1 = arrs0[-1].split('_')
                        if arrs1[0].startswith('白') or arrs1[0].startswith('夜'):
                            continue
                        arrs2 = arrs1[0].split('#')
                        ggh = arrs2[0]
                        print('ggh: {0};'.format(ggh))