# 品牌车型年款控制器类
from flask import request
from apps.admin.controller.flask_web import FlaskWeb
from apps.admin.controller.c_brand import CBrand
from apps.admin.controller.c_model import CModel
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_bmy import MBmy
from apps.admin.model.m_bmy_example import MBmyExample
from apps.admin.model.m_data_source import MDataSource

class CBmy(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBmy'

    @staticmethod
    def add_bmy(brand_id, model_id, bmy_name):
        bmy_vo = MBmy.get_bmy_by_name(bmy_name)
        if bmy_vo is None:
            bmy_id = MPkGenerator.get_pk('bmy')
            bmy_vo = {
                'brand_id': brand_id,
                'model_id': model_id,
                'bmy_id': bmy_id,
                'bmy_name': bmy_name,
                'bmy_pics': 0
            }
            MBmy.insert(bmy_vo)
        return bmy_vo['bmy_id']

    @staticmethod
    def get_bmy_by_name(bmy_name):
        return MBmy.get_bmy_by_name(bmy_name)

    @staticmethod
    def find_bmy_id_by_name(bmy_name):
        '''
        如果t_bmy表中有相应记录，则不做任何处理；如果不存在，则
        依次添加品牌、车型、年款到相应数据库表
        参数：bmy_name 品牌车型年款名称
        返回值：bmy_id
        '''
        bmy_vo = MBmy.get_bmy_by_name(bmy_name)
        arrs0 = bmy_name.split('_')
        brand_name = arrs0[0]
        model_name = '{0}_{1}'.format(arrs0[0], arrs0[1])
        if bmy_vo is None:
            brand_id = CBrand.add_brand(brand_name)
            model_id = CModel.add_model(brand_id, model_name)
            bmy_id = CBmy.add_bmy(brand_id, model_id, bmy_name)
        else:
            bmy_id = bmy_vo['bmy_id']
        return bmy_id

    @staticmethod
    def get_bmy_ids():
        return MBmy.get_bmy_ids()

    @staticmethod
    def get_bmys_api():
        bmys = MBmy.get_bmys()
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'total': len(bmys),
            'bmys': bmys
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def get_bmys():
        return MBmy.get_bmys()

    @staticmethod
    def get_bmy_example_vehicle_image_id_api():
        '''
        获取指定品牌车型年款示例图像编号，如果没有返回0
        '''
        bmy_id = request.args.get("bmyId")
        vehicle_image_id = CBmy.get_bmy_example_vehicle_image_id(bmy_id)
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'vehicle_image_id': vehicle_image_id
        }
        return FlaskWeb.generate_response(resp_param)

    @staticmethod
    def get_bmy_example_vehicle_image_id(bmy_id):
        rec = MBmyExample.get_bmy_example_vehicle_image_id(int(bmy_id))
        if len(rec) < 1:
            return 0
        else:
            return int(rec['vehicle_image_id'])

    @staticmethod
    def get_bmy_current_vehicle_image_id_api():
        bmy_id = int(request.args.get("bmyId"))
        prev_vehicle_id = int(request.args.get("prevVehicleImageId"))
        mode = int(request.args.get('mode'))
        vehicle_image_id = CBmy.get_bmy_current_vehicle_image_id(bmy_id, prev_vehicle_id, mode)
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'vehicle_image_id': vehicle_image_id
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def get_bmy_current_vehicle_image_id(bmy_id, prev_vehicle_id, mode):
        rec = MDataSource.get_bmy_current_vehicle_image_id(bmy_id, prev_vehicle_id, mode)
        if len(rec) < 1:
            return 0
        else:
            return int(rec['vehicle_image_id'])

    @staticmethod
    def set_bmy_example_vehicle_image_id_api():
        bmy_id = int(request.args.get("bmyId"))
        vehicle_image_id = int(request.args.get("vehicleImageId"))
        CBmy.set_bmy_example_vehicle_image_id(bmy_id, vehicle_image_id)
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'result': 'Ok'
        }
        return FlaskWeb.generate_response(resp_param)

    @staticmethod
    def set_bmy_example_vehicle_image_id(bmy_id, vehicle_image_id):
        bmy_example_id = MPkGenerator.get_pk('bmy_example')
        MBmyExample.set_bmy_example_vehicle_image_id(bmy_example_id, bmy_id, vehicle_image_id)

    @staticmethod
    def process_brand_rename(old_brand_name, new_brand_name):
        MBmy.process_brand_rename(old_brand_name, new_brand_name)

    @staticmethod
    def add_bmy_brand_name_postfix():
        '''
        在品牌车型年款名称的品牌名称后面加下牌
        '''
        MBmy.add_bmy_brand_name_postfix()

