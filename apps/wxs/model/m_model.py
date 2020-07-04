# 
import pymongo

class MModel(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MModel'

    @staticmethod
    def is_model_exists(model_name):
        if MModel.db is None:
            MModel._initialize()
        query_cond = {'model_name': model_name}
        fields = {'model_id': 1, 'model_name': 1, 'model_num': 1}
        if MModel.tbl.find_one(query_cond, fields) is None:
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
        if MModel.db is None:
            MModel._initialize()
        return MModel.tbl.insert_one(model_vo)





    @staticmethod
    def get_brand_by_name(brand_name):
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'brand_name': brand_name}
        fields = {'brand_id': 1, 'brand_name': 1, 'brand_num': 1}
        return MBrand.tbl.find_one(query_cond, fields)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MModel.db = mongo_client['stpdb']
        MModel.tbl = MModel.db['t_model']