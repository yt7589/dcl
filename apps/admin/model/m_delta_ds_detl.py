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
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDeltaDsDetl.db = mongo_client['tcvdb']
        MDeltaDsDetl.tbl = MDeltaDsDetl.db['t_delta_ds_detl']