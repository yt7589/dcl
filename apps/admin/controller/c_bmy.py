# 车型（品牌_车型_年款）
import json
from pathlib import Path
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
        mode = request.args.get("mode")
        if '1' == mode:
            bmys = CBmy.get_bmys_from_db()
        elif '2' == mode:
            bmys = CBmy.get_bmys_from_folder()
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'total': len(bmys),
            'bmys': bmys
        }
        return FlaskWeb.generate_response(resp_param)

    @staticmethod
    def get_bmys_from_db():
        pass

    @staticmethod
    def get_bmys_from_folder():
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        raw_bmys = []
        stop_loop = False
        sl_num = 0
        for brand_path in base_path.iterdir():
            if stop_loop:
                break
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            for model_path in brand_path.iterdir():
                if stop_loop:
                    break
                model_str = str(model_path)
                arrs1 = model_str.split('/')
                model_name = arrs1[-1]
                for year_path in model_path.iterdir():
                    if stop_loop:
                        break
                    year_str = str(year_path)
                    arrs2 = year_str.split('/')
                    year_name = arrs2[-1]
                    num = 0
                    for img_file in year_path.iterdir():
                        num += 1
                    bmy_name = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                    bmy = {'bmy_name': bmy_name, 'bmy_num': num}
                    raw_bmys.append(bmy)
                    if sl_num > 10:
                        stop_loop = True
                    sl_num += 1
        bmys = []
        for rb in raw_bmys:
            print('### {0};'.format(rb))
        recs = sorted(raw_bmys, key=CBmy.sort_by_num_bmy, reverse=False)
        bmy_id = 1
        for rec in recs:
            print('@@@ {0};'.format(rec))
            rec['bmy_id'] = bmy_id
            bmy = {
                'bmy_id': rec['bmy_id'],
                'bmy_name': rec['bmy_name'],
                'bmy_num': rec['bmy_num']
            }
            bmy_id += 1
            bmys.append(bmy)
        return bmys

    @staticmethod
    def sort_by_num_bmy(item):
        return '{0:10d}_{1}'.format(item['bmy_num'], item['bmy_name'])