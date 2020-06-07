
import json
from pathlib import Path
#import flask
#from flask import Flask, jsonify
#from flask_cors import CORS
from flask import request
from apps.admin.controller.flask_web import FlaskWeb

class CGghBmy(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CGghBmy'

    @staticmethod
    def ggh_to_bmy_dict_api():
        rst = CGghBmy.ggh_to_bmy_dict()
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'ggh_num': rst.ggh_num,
            'brand_num': rst.brand_num,
            'model_num': rst.model_num,
            'bmy_num': rst.bmy_num
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def ggh_to_bmy_dict():
        rst = {
            'ggh_num': 201,
            'brand_num': 202,
            'model_num': 203,
            'bmy_num': 104
        }
        return rst