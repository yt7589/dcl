#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_bmy import MBmy
from apps.wxs.model.m_vin import MVin

class CBmy(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CBmy'

    @staticmethod
    def is_bmy_exists(bmy_name):
        return MBmy.is_bmy_exists(bmy_name)

    @staticmethod
    def add_bmy(bmy_name, bmy_code, brand_id, brand_code, model_id, model_code):
        if MBmy.is_bmy_exists(bmy_name):
            bmy_vo = MBmy.get_bmy_by_name(bmy_name)
            return int(bmy_vo['bmy_id'])
        bmy_id = MPkGenerator.get_pk('bmy_id')
        bmy_vo = {
            'bmy_id': bmy_id,
            'bmy_name': bmy_name,
            'bmy_code': bmy_code,
            'brand_id': brand_id,
            'brand_code': brand_code,
            'model_id': model_id,
            'model_code': model_code,
            'bmy_num': 1
        }
        rst = MBmy.insert(bmy_vo)
        return bmy_id

    @staticmethod
    def is_vin_exists(vin_code):
        return MVin.is_vin_exists(vin_code)

    @staticmethod
    def add_vin(vin_code, bmy_id, source_type):
        if MVin.is_vin_exists(vin_code):
            vin_vo = MVin.get_vin_by_code(vin_code)
            return int(vin_vo['vin_id'])
        vin_id = MPkGenerator.get_pk('vin_id')
        vin_vo = {
            'vin_id': vin_id,
            'vin_code': vin_code,
            'bmy_id': bmy_id,
            'source_type': source_type
        }
        rst = MVin.insert(vin_vo)
        return vin_id

    @staticmethod
    def get_bmy_id_by_vin_code(vin_code):
        rec = MVin.get_bmy_id_by_vin_code(vin_code)
        if rec:
            return int(rec['bmy_id'])
        else: 
            return -1


    