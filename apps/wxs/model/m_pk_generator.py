# 维护所有表的主键值
import pymongo

class MPkGenerator(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MSeq'

    @staticmethod
    def get_pk(pk_name):
        if MPkGenerator.db is None:
            MPkGenerator._initialize()
        query_cond = {'pk_name': pk_name}
        fields = {'pk_val': 1}
        rec = MPkGenerator.tbl.find_one(query_cond, fields)
        if rec is None:
            # 添加记录
            MPkGenerator.tbl.insert_one({
                'pk_name': pk_name,
                'pk_val': 1
            })
            rec = MPkGenerator.tbl.find_one(query_cond, fields)
        pk_val = rec['pk_val']
        MPkGenerator.tbl.update_one(query_cond, {'$set': {'pk_val': pk_val+1}})
        return int(pk_val)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MPkGenerator.db = mongo_client['stpdb']
        MPkGenerator.tbl = MPkGenerator.db['t_pk']