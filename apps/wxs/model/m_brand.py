# 
import pymongo

class MBrand(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MBrand'

    @staticmethod
    def is_brand_exists(brand_name):
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        if MBrand.tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def insert(brand_vo):
        '''
        向t_brand表中添加记录，brand_vo中包括：
            brand_id, brand_name, brand_code, brand_num=1
        '''
        if MBrand.db is None:
            MBrand._initialize()
        return MBrand.tbl.insert_one(brand_vo)

    @staticmethod
    def get_brand_by_name(brand_name):
        '''
        根据品牌名称求出品牌详细信息
        '''
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_code': 1, 
                    'source_type': 1, 'brand_num': 1}
        return MBrand.tbl.find_one(query_cond, fields)




    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBrand.db = mongo_client['stpdb']
        MBrand.tbl = MBrand.db['t_brand']