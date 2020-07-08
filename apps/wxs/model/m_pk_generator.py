# 维护所有表的主键值
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MPkGenerator(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MSeq'

    @staticmethod
    def get_pk(pk_name):
        query_cond = {'pk_name': pk_name}
        fields = {'pk_val': 1}
        tbl = MMongoDb.db['t_pk']
        rec = tbl.find_one(query_cond, fields)
        if rec is None:
            # 添加记录
            tbl.insert_one({
                'pk_name': pk_name,
                'pk_val': 1
            })
            rec = tbl.find_one(query_cond, fields)
        pk_val = rec['pk_val']
        tbl.update_one(query_cond, {'$set': {'pk_val': pk_val+1}})
        return int(pk_val)