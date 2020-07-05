# 
import pymongo

class MVin(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MVin'

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MVin.db = mongo_client['stpdb']
        MVin.tbl = MVin.db['t_vin']