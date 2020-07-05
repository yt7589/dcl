# 
import pymongo

class MBmy(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MBmy'

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmy.db = mongo_client['stpdb']
        MBmy.tbl = MBmy.db['t_bmy']