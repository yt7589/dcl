# 车型（品牌_车型_年款）
import json
#import flask
#from flask import Flask, jsonify
#from flask_cors import CORS
from flask import request
from apps.admin.controller.flask_web import FlaskWeb

class CBmy(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBmy'

    @staticmethod
    def get_bmys_api():
        userId = request.args.get("userId")
        print('userId={0};'.format(userId))
        resp_param = FlaskWeb.get_resp_param()
        return FlaskWeb.generate_response()