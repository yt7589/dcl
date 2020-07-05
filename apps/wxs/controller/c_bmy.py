#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_bmy import MBmy

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

    