
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
        for k, v in ggh_to_bmy_dict.items():
            print('{0}:{1};'.format(k, v))
        rst = {
            'ggh_num': 201,
            'brand_num': 202,
            'model_num': 203,
            'bmy_num': 104
        }
        return rst