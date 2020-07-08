# MongoDB的公共类
import pymongo

class MMongoDb(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MMongoDb'

    @staticmethod
    def convert_recs(recs):
        '''
        将具有多条记录的cursor转为list[dict]
        '''
        rows = []
        for rec in recs:
            row = {}
            for k, v in rec.items():
                if k != '_id':
                    row[k] = v
            rows.append(row)
        return rows

    @staticmethod
    def convert_rec(rec):
        '''
        将只有一条的结果转为dict
        '''
        row = {}
        if rec is None:
            return row
        for k, v in rec.items():
            if k != '_id':
                row[k] = v
        return row

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MMongoDb.db = mongo_client['stpdb']