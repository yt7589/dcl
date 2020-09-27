# 
from os import stat
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MModel(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MModel'

    @staticmethod
    def is_model_exists(model_name):
        query_cond = {'model_name': model_name}
        fields = {'model_id': 1, 'model_name': 1, 'model_num': 1}
        if MMongoDb.db['t_model'].find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def insert(model_vo):
        '''
        向t_model表中添加记录，model_vo中包括：
            model_id, model_name, model_code, brand_id, 
            brand_code, model_num=1
        '''
        return MMongoDb.db['t_model'].insert_one(model_vo)

    @staticmethod
    def get_model_by_name(model_name):
        query_cond = {'model_name': model_name}
        fields = {'model_id': 1, 'model_name': 1, 'model_code': 1, 'model_num': 1}
        return MMongoDb.db['t_model'].find_one(query_cond, fields)

    @staticmethod
    def get_model_vo_by_id(model_id):
        query_cond = {'model_id': model_id}
        fields = {'model_name': 1, 'model_code': 1, 'source_type': 1}
        return MMongoDb.db['t_model'].find_one(query_cond, fields)
        
    @staticmethod
    def get_wxs_bms():
        query_cond = {'source_type': 1}
        fields = {'model_code':1, 'model_name': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_model']\
                    .find(query_cond, fields))