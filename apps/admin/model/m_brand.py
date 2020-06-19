# 
import pymongo

class MBrand(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBrand'

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
    def get_brand_by_name(brand_name):
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        return MBrand.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(brand_vo):
        if MBrand.db is None:
            MBrand._initialize()
        rst = MBrand.tbl.insert_one(brand_vo)

    @staticmethod
    def query_brands(start_idx=1, amount=-1, sort_id=1,
                 sort_type=1):
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        return MBrand.tbl.find(query_cond, fields)

    @staticmethod
    def clear_brands():
        if MBrand.db is None:
            MBrand._initialize()
        MBrand.tbl.delete_many({})

    @staticmethod
    def add_brand_name_postfix():
        ''' 
        在品牌名称后面加牌
        '''
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'brand_id': {'$gt': 0}}
        fields = {'brand_id': 1, 'brand_name': 1}
        recs = MBrand.tbl.find(query_cond, fields).sort('brand_id', 1)
        for rec in recs:
            old_brand_name = rec['brand_name']
            new_brand_name = '{0}牌'.format(old_brand_name)
            print('编号{0}: {1} => {2}'.format(rec['brand_id'], old_brand_name, new_brand_name))

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBrand.db = mongo_client['tcvdb']
        MBrand.tbl = MBrand.db['t_brand']