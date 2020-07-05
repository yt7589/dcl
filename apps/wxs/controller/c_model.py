#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_model import MModel

class CModel(object):
    def __init__(self):
        self.name = 'apps.controller.CModel'

    @staticmethod
    def add_model(model_name, model_code, brand_vo, source_type):
        if MModel.is_model_exists(model_name):
            return
        model_id = MPkGenerator.get_pk('model_id')
        model_vo = {
            'model_id': model_id,
            'model_name': model_name,
            'model_code': model_code,
            'brand_id': brand_vo['brand_id'],
            'source_type': source_type
        }
        rst = MModel.insert(model_vo)
        return True

    @staticmethod
    def get_model_by_name(model_name):
        return MModel.get_model_by_name(model_name)
        