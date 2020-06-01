# 车型（品牌_车型_年款）
import json
from pathlib import Path
#import flask
#from flask import Flask, jsonify
#from flask_cors import CORS
from flask import request
from apps.admin.controller.flask_web import FlaskWeb
from apps.admin.model.m_bmys import MBmys

class CBmys(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBmy'

    @staticmethod
    def get_bmys_api():
        mode = request.args.get("mode")
        if '1' == mode:
            bmys = CBmy.get_bmys_from_db()
        elif '2' == mode:
            bmys = CBmys.get_bmys_from_folder()
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
        MBmys.delete_all()
        raw_bmys = []
        for brand_path in base_path.iterdir():
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            for model_path in brand_path.iterdir():
                model_str = str(model_path)
                arrs1 = model_str.split('/')
                model_name = arrs1[-1]
                for year_path in model_path.iterdir():
                    year_str = str(year_path)
                    arrs2 = year_str.split('/')
                    year_name = arrs2[-1]
                    num = 0
                    for img_file in year_path.iterdir():
                        num += 1
                    bmy_name = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                    bmy = {'bmy_name': bmy_name, 'bmy_num': num}
                    raw_bmys.append(bmy)
        bmys = []
        recs = sorted(raw_bmys, key=CBmys.sort_by_num_bmy, reverse=False)
        bmy_id = 1
        for rec in recs:
            rec['bmy_id'] = bmy_id
            bmy = {
                'bmy_id': rec['bmy_id'],
                'bmy_name': rec['bmy_name'],
                'bmy_num': rec['bmy_num']
            }
            bmy_id += 1
            bmys.append(bmy)
        for bmy in bmys:
            rec = {
                'bmy_id': bmy['bmy_id'],
                'bmy_name': bmy['bmy_name'],
                'bmy_num': bmy['bmy_num']
            }
            MBmys.insert(rec)
        return bmys

    @staticmethod
    def sort_by_num_bmy(item):
        return '{0:10d}_{1}'.format(item['bmy_num'], item['bmy_name'])