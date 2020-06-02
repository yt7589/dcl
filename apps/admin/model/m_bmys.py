# 处理bmy表
import pymongo

class MBmys(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBmys'

    @staticmethod
    def insert(bmy_vo):
        if MBmys.tbl is None:
            MBmys._initialize()
        MBmys.tbl.insert_one(bmy_vo)

    @staticmethod
    def delete_all():
        if MBmys.tbl is None:
            MBmys._initialize()
        MBmys.tbl.delete_many({})

    @staticmethod
    def query_bmys():
        if MBmys.tbl is None:
            MBmys._initialize()
        query_cond = {}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_num': 1}
        return MBmys.tbl.find(query_cond, fields)
    
    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmys.db = mongo_client['tcvdb']
        MBmys.tbl = MBmys.db['bmys']