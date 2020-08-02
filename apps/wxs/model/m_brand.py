# 
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MBrand(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MBrand'

    @staticmethod
    def is_brand_exists(brand_name):
        tbl = MMongoDb.db['t_brand']
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        if tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def insert(brand_vo):
        '''
        向t_brand表中添加记录，brand_vo中包括：
            brand_id, brand_name, brand_code, brand_num=1
        '''
        return MMongoDb.db['t_brand'].insert_one(brand_vo)

    @staticmethod
    def get_brand_by_name(brand_name):
        '''
        根据品牌名称求出品牌详细信息
        '''
        tbl = MMongoDb.db['t_brand']
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_code': 1, 
                    'source_type': 1, 'brand_num': 1}
        return tbl.find_one(query_cond, fields)

    @staticmethod
    def get_wxs_brands():
        query_cond = {'source_type': 1}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_code': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_brand']\
            .find(query_cond, fields))

    @staticmethod
    def get_brand_vo_by_id(brand_id):
        query_cond = {'brand_id': brand_id}
        fields = {'brand_id': 1, 'brand_name': 1, 
                    'brand_code': 1, 'source_type': 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_brand']\
            .find(query_cond, fields))