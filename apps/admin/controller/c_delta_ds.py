# 增量数据集管理类
import time
from flask import request
from apps.admin.controller.flask_web import FlaskWeb
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_delta_ds import MDeltaDs
from apps.admin.model.m_delta_ds_detl import MDeltaDsDetl
from apps.admin.model.m_data_source import MDataSource
from apps.admin.model.m_bmy import MBmy

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
    def get_worker_delta_ds_detl_api():
        worker_id = request.args.get('workerId')
        delta_ds_id = int(request.args.get('deltaDsId'))
        delta_ds_detl_id = int(request.args.get('deltaDsDetlId'))
        mode = int(request.args.get('mode'))
        delta_ds_detl_vo = CDeltaDs.get_worker_delta_ds_detl(worker_id, delta_ds_id, delta_ds_detl_id, mode)
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'delta_ds_detl_vo': delta_ds_detl_vo
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def get_worker_delta_ds_detl(worker_id, delta_ds_id, delta_ds_detl_id, mode):
        '''
        '''
        # 根据worker_id求出delta_ds_id
        if delta_ds_id < 1:
            rec = MDeltaDs.get_work_delta_ds_id(worker_id)
            if len(rec) < 1:
                return []
            delta_ds_id = rec['delta_ds_id']
        if delta_ds_id < 1:
            return {}
        rec = MDeltaDsDetl.get_delta_ds_detl(delta_ds_id, delta_ds_detl_id, mode)
        rec['data_source_id'] = int(rec['data_source_id'])
        bmys = MBmy.get_bmys()
        rec['bmy_name'] = bmys[rec['bmy_id']]['bmy_name']
        delta_ds_vo = MDataSource.get_vo_by_data_source_id(rec['data_source_id'])
        rec['vehicle_image_id'] = int(delta_ds_vo['vehicle_image_id'])
        return rec

    @staticmethod
    def set_delta_ds_detl_state_api():
        delta_ds_detl_id = int(request.args.get('deltaDsDetlId'))
        state = int(request.args.get('state'))
        delta_ds_detl_vo = CDeltaDs.set_delta_ds_detl_state(delta_ds_detl_id, state)
        resp_param = FlaskWeb.get_resp_param()
        resp_param['data'] = {
            'delta_ds_detl_vo': delta_ds_detl_vo
        }
        return FlaskWeb.generate_response(resp_param)
    @staticmethod
    def set_delta_ds_detl_state(delta_ds_detl_id, state):
        '''
        设置增量数据集t_delta_ds_detl表中对应记录状态
        '''
        MDeltaDsDetl.set_delta_ds_detl_state(delta_ds_detl_id, state)

    @staticmethod
    def get_check_delta_ds_detls(worker_id):
        rec = MDeltaDs.get_work_delta_ds_id(worker_id)
        if len(rec) < 1:
            return []
        delta_ds_id = rec['delta_ds_id']
        print('delta_ds_id={0};'.format(delta_ds_id))
        # 随机抽取当天state=1的记录
        sample_num = 3
        recs = MDeltaDsDetl.get_worker_normal_delta_ds_detls(delta_ds_id, sample_num)
        for rec in recs:
            print(rec)
        # 取所有state=2或3的记录