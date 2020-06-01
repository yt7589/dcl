# 
import pymongo

class MBrand(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBrand'

    @staticmethod
    def insert(brand_vo):
        if MBrand.db is None:
            MBrand.initialize()
        rst = MBrand.tbl.insert_one(brand_vo)

    @staticmethod
    def query_brands(start_idx=1, amount=-1, sort_id=1,
                 sort_type=1):
        if MBrand.db is None:
            MBrand.initialize()
        query_cond = {}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        return MBrand.tbl.find(query_cond, fields)

    @staticmethod
    def initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBrand.db = mongo_client['tcvdb']
        MBrand.tbl = MBrand.db['brands']