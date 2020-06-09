# 增量数据集主表模型类
import pymongo
from apps.admin.model.m_mongodb import MMongoDb

class MDeltaDs(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MDeltaDs'
        
    @staticmethod
    def insert(delta_ds_vo):
        if MDeltaDs.db is None:
            MDeltaDs._initialize()
        return MDeltaDs.tbl.insert_one(delta_ds_vo)

    @staticmethod
    def delete_delta_ds(delta_ds_id):
        if MDeltaDs.db is None:
            MDeltaDs._initialize()
        return MDeltaDs.tbl.delete_one({'delta_ds_id': delta_ds_id})

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDeltaDs.db = mongo_client['tcvdb']
        MDeltaDs.tbl = MDeltaDs.db['t_delta_ds']