# 增量数据集管理类
import time
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_delta_ds import MDeltaDs
from apps.admin.model.m_delta_ds_detl import MDeltaDsDetl

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

    @staticmethod
    def add_delta_ds_detl(delta_ds_id, data_source_id, image_full_path, bmy_id):
        delta_ds_detl_id = MPkGenerator.get_pk('delta_ds_detl')
        delta_ds_detl_vo = {
            'delta_ds_detl_id': delta_ds_detl_id,
            'delta_ds_id': delta_ds_id,
            'data_source_id': data_source_id,
            'image_full_path': image_full_path,
            'bmy_id': bmy_id,
            'state': 4,
            'last_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        MDeltaDsDetl.insert(delta_ds_detl_vo)
        return delta_ds_detl_id