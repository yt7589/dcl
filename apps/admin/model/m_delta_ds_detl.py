# 增量数据集明细表
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
        fields = {'data_source_id': 1, 'state': 1, 'vehicle_image_id': 1, 'bmy_id': 1, 'delta_ds_detl_id': 1}
        return MMongoDb.convert_recs(MDeltaDsDetl.tbl.find(query_cond, fields))

    @staticmethod
    def delete_delta_ds_detls(delta_ds_id):
        if MDeltaDsDetl.db is None:
            MDeltaDsDetl._initialize()
        return MDeltaDsDetl.tbl.delete_many({'delta_ds_id': delta_ds_id})

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDeltaDsDetl.db = mongo_client['tcvdb']
        MDeltaDsDetl.tbl = MDeltaDsDetl.db['t_delta_ds_detl']