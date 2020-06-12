# 增量数据集管理类
import time
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_delta_ds import MDeltaDs
from apps.admin.model.m_delta_ds_detl import MDeltaDsDetl
from apps.admin.model.m_data_source import MDataSource

class CDeltaDs(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CDeltaDs'

    @staticmethod
    def create_delta_ds(worker_id):
        '''
        在t_delta_ds表中生成新记录
        '''
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

    @staticmethod
    def save_to_dataset(delta_ds_id):
        '''
        将t_delta_ds_detl表中增量数据集中记录的状态更新到t_data_source表中
        '''
        delta_ds_detls = MDeltaDsDetl.get_delta_ds_detls(delta_ds_id)
        for rec in delta_ds_detls:
            MDataSource.update_state(rec['data_source_id'], int(rec['state']))
            print('更新{0}状态为{1};'.format(rec['data_source_id'], int(rec['state'])))
        CDeltaDs.delete_delta_ds(delta_ds_id)

    @staticmethod
    def delete_delta_ds(delta_ds_id):
        MDeltaDsDetl.delete_delta_ds_detls(delta_ds_id)
        MDeltaDs.delete_delta_ds(delta_ds_id)

    @staticmethod
    def get_worker_delta_ds_detls(worker_id):
        # 根据worker_id求出delta_ds_id
        print('求出delta_ds_id')
        # 求出delta_ds_detls
        # 加入bmy_name