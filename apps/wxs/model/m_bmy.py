# 
from os import stat
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MBmy(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MBmy'

    @staticmethod
    def is_bmy_exists(bmy_name):
        tbl = MMongoDb.db['t_bmy']
        query_cond = {'bmy_name': bmy_name}
        fields = {'bmy_id': 1, 'bmy_num': 1}
        if tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_bmy_by_name(bmy_name):
        tbl = MMongoDb.db['t_bmy']
        query_cond = {'bmy_name': bmy_name}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                    'brand_id': 1, 'brand_code': 1,
                    'model_id': 1, 'model_code': 1,
                    'model_num': 1}
        return tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(bmy_vo):
        '''
        向t_bmy表中添加记录，model_vo中包括：
            bmy_id, bmy_name, bmy_code, brand_id, brand_code, 
            model_id, model_code, is_imported_vehicle, model_num=1
        '''
        return MMongoDb.db['t_bmy'].insert_one(bmy_vo)

    @staticmethod
    def get_bmy_by_id(bmy_id):
        query_cond = {'bmy_id': bmy_id}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                    'brand_id': 1, 'brand_code': 1,
                    'model_id': 1, 'model_code': 1,
                    'model_num': 1}
        return MMongoDb.db['t_bmy'].find_one(query_cond, fields)

    @staticmethod
    def get_bmy_id_bmy_names():
        query_cond = {}
        fields = {'bmy_id': 1, 'bmy_name': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy'].find(query_cond, fields))

    @staticmethod
    def get_bmy_id_bmy_vos():
        query_cond = {}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                    'brand_id': 1, 'brand_code': 1, 'model_id': 1, 
                    'model_code': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields))

    @staticmethod
    def get_bmys():
        query_cond = {}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                    'brand_id': 1, 'brand_code': 1, 'model_id': 1, 
                    'model_code': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields).sort([('bmy_id', pymongo.ASCENDING)]))

    @staticmethod
    def get_bmy_id_brand_ids():
        query_cond = {}
        fields = {'bmy_id': 1, 'brand_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields))

    @staticmethod
    def get_bmy_id_model_ids():
        '''
        获取bmy_id对应model_id的对应关系
        '''
        query_cond = {}
        fields = {'bmy_id': 1, 'model_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields))

    @staticmethod
    def get_bmy_code_to_bmy_id_dict():
        '''
        获取bmy_code到bmy_id字典
        '''
        query_cond = {}
        fields = {'bmy_code': 1, 'bmy_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_bmy']\
                    .find(query_cond, fields))

    @staticmethod
    def update_bmy_codes(bmy_id, bmy_code, bm_code, brand_code):
        query_cond = {'bmy_id': bmy_id}
        fields = {'$set': {
            'bmy_code': bmy_code, 'model_code': bm_code, 
            'brand_code': brand_code
        }}
        MMongoDb.db['t_bmy'].update_one(query_cond, fields)