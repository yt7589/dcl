# 品牌车型年款控制器类
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_bmy import MBmy

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