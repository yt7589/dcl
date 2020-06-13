# 增量数据集明细表
import time
import random
import pymongo
from apps.admin.model.m_mongodb import MMongoDb

class MDeltaDsDetl(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MDeltaDsDetl'

    @staticmethod
    def insert(delta_ds_detl_vo):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        return MDeltaDsDetl.tbl.insert_one(delta_ds_detl_vo)

    @staticmethod
    def get_delta_ds_detls(delta_ds_id):
        '''
        '''
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        query_cond = {'delta_ds_detl_id': {'$gt': 0}}
        fields = {'data_source_id': 1, 'state': 1}
        return MMongoDb.convert_recs(MDeltaDsDetl.tbl.find(query_cond, fields))

    @staticmethod
    def get_delta_ds_detl(delta_ds_id, delta_ds_detl_id, mode):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        if mode == 1:
            query_cond = {'delta_ds_id': delta_ds_id, 'delta_ds_detl_id': {'$gt': delta_ds_detl_id}}
        else:
            query_cond = {'delta_ds_id': delta_ds_id, 'delta_ds_detl_id': {'$lt': delta_ds_detl_id}}
        fields = {'delta_ds_detl_id': 1, 'data_source_id': 1, 'bmy_id': 1}
        return MMongoDb.convert_rec(MDeltaDsDetl.tbl.find_one(query_cond, fields))

    @staticmethod
    def delete_delta_ds_detls(delta_ds_id):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        return MDeltaDsDetl.tbl.delete_many({'delta_ds_id': delta_ds_id})

    @staticmethod
    def set_delta_ds_detl_state(delta_ds_detl_id, state):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        last_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        query_cond = {'delta_ds_detl_id': delta_ds_detl_id}
        new_values = {'$set': {'state': state, 'last_date': last_date}}
        MDeltaDsDetl.tbl.update_one(query_cond, new_values)

    @staticmethod
    def get_worker_normal_delta_ds_detls(delta_ds_id, sample_num):
        '''
        从选择正常的记录中，随机采样出指定数量记录用于质量抽查
        '''
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        last_date = time.strftime("%Y-%m-%d", time.localtime())
        regex_cond = '^{0}'.format(last_date)
        query_cond = {'delta_ds_id': delta_ds_id, 'last_date': {'$regex': regex_cond}, 'state': 1}
        fields = {'delta_ds_detl_id': 1, 'data_source_id': 1, 'bmy_id': 1}
        rows = MMongoDb.convert_recs(MDeltaDsDetl.tbl.find(query_cond, fields))
        cnt = len(rows)
        if cnt <= sample_num:
            return rows
        recs = []
        for idx in range(sample_num):
            random_num = random.randint(0, cnt-1)
            recs.append(rows[idx])
        return recs

    @staticmethod
    def get_worker_abnormal_delta_ds_detls(delta_ds_id):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        query_cond = {'delta_ds_id': delta_ds_id, 'last_date': 
                    {'$regex': regex_cond}, 
                    '$or': [{'state': 2}, {'state': 3}]}
        fields = {'delta_ds_detl_id': 1, 'data_source_id': 1, 'bmy_id': 1}
        return MMongoDb.convert_recs(MDeltaDsDetl.tbl.find(query_cond, fields))

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDeltaDsDetl.db = mongo_client['tcvdb']
        MDeltaDsDetl.tbl = MDeltaDsDetl.db['t_delta_ds_detl']