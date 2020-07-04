# Flask框架基础方法定义
import json

class FlaskWeb(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.FlaskWeb'

    @staticmethod
    def get_resp_param():
        return {
            'code': 0,
            'msg': 'Ok',
            'data': []
        }

    @staticmethod
    def generate_response(resp_param):
        resp = {
            'code': resp_param['code'],
            'msg': resp_param['msg'],
            'data': resp_param['data']
        }
        return json.dumps(resp, ensure_ascii=False)