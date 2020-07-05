# 
import pymongo

class MBmy(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MBmy'

    @staticmethod
    def is_bmy_exists(bmy_name):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_name': bmy_name}
        fields = {'bmy_id': 1, 'bmy_num': 1}
        if MBmy.tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_bmy_by_name(bmy_name):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_name': bmy_name}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_code': 1, 
                    'brand_id': 1, 'brand_code': 1,
                    'model_id': 1, 'model_code': 1,
                    'model_num': 1}
        return MBmy.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(bmy_vo):
        '''
        向t_bmy表中添加记录，model_vo中包括：
            bmy_id, bmy_name, bmy_code, brand_id, brand_code, 
            model_id, model_code, is_imported_vehicle, model_num=1
        '''
        if MBmy.db is None:
            MBmy._initialize()
        return MBmy.tbl.insert_one(bmy_vo)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmy.db = mongo_client['stpdb']
        MBmy.tbl = MBmy.db['t_bmy']