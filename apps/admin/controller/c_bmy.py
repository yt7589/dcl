# 车型（品牌_车型_年款）
import json
#import flask
#from flask import Flask, jsonify
#from flask_cors import CORS
from flask import request

class CBmy(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBmy'

    '''
    @app.route('/admin/getBmys', methods=['GET'])
    @staticmethod
    def get_bmys_api():
        userId = request.args.get("userId")
        resp = {
            'code': 0,
            'msg': 'Ok',
            'data': data
        }
        return json.dumps(resp, ensure_ascii=False)
    '''