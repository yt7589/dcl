# 
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MDataset(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MDataset'

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDataset.db = mongo_client['stpdb']
        MDataset.tbl = MBmy.db['t_dataset']