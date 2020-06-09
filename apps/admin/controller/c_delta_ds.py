# 增量数据集管理类
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_delta_ds import MDeltaDs

class CDeltaDs(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CDeltaDs'

    @staticmethod
    def create_delta_ds():
        '''
        在t_delta_ds表中生成新记录
        '''
        worker_id = 1
        delta_ds_id = MPkGenerator.get_pk('delta_ds')
        delta_ds_vo = {
            'delta_ds_id': delta_ds_id,
            'worker_id': worker_id,
            'state': 0
        }
        MDeltaDs.insert(delta_ds_vo)
        return delta_ds_id