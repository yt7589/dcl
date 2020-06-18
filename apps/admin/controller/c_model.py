# 车型控制器类
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_model import MModel

class CModel(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CModel'

    @staticmethod
    def add_model(brand_id, model_name):
        model_vo = MModel.get_model_by_name(model_name)
        if model_vo is None:
            model_id = MPkGenerator.get_pk('model')
            model_vo = {
                'model_id': model_id,
                'brand_id': brand_id,
                'model_name': model_name,
                'model_num': 0
            }
            MModel.insert(model_vo)
        return model_vo['model_id']

    @staticmethod
    def process_tesla_rename():
        MModel.process_tesla_rename()