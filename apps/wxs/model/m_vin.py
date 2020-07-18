# 
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MVin(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MVin'

    @staticmethod
    def is_vin_exists(vin_code):
        query_cond = {'vin_code': vin_code}
        fields = {'vin_id': 1, 'bmy_id': 1}
        if MMongoDb.db['t_vin'].find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_vin_by_code(vin_code):
        query_cond = {'vin_code': vin_code}
        fields = {'vin_id': 1, 'vin_code': 1, 'bmy_id': 1, 
                    'source_type': 1}
        return MMongoDb.db['t_vin'].find_one(query_cond, fields)

    @staticmethod
    def insert(vin_vo):
        '''
        向t_vin表中添加记录，vin_vo中包括：
            vin_id, vin_code, bmy_id, source_type
        '''
        return MMongoDb.db['t_vin'].insert_one(vin_vo)

    @staticmethod
    def get_bmy_id_by_vin_code(vin_code):
        query_cond = {"vin_code": vin_code}
        fields = {"vin_id": 1, "bmy_id": 1}
        return MMongoDb.convert_rec(MMongoDb.db['t_vin'].find_one(query_cond, fields))

    @staticmethod
    def get_bmy_ids_by_vin_code(prefix_vin_code):
        query_cond = {"vin_code": prefix_vin_code}
        fields = {"vin_id": 1, "bmy_id": 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_vin'].find(query_cond, fields))

    @staticmethod
    def get_vin_bmy_id_dict():
        query_cond = {}
        fields = {'vin_code': 1, 'bmy_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_vin'].find(query_cond, fields))

    @staticmethod
    def get_vin_codes():
        query_cond = {}
        fields = {"vin_id": 1, "vin_code": 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_vin'].find(query_cond, fields).sort([('vin_code', 1)]))

    @staticmethod
    def get_vin_code_bmy_id_dict():
        query_cond = {}
        fields = {'vin_code': 1, 'bmy_id': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_vin']\
                    .find(query_cond, fields))