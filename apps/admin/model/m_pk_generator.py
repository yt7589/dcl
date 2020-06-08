# 
import pymongo

class MPkGenerator(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MPkGenerator'

    @staticmethod
    def get_pk(pk_name):
        if MBrand.db is None:
            MBrand._initialize()
        query_cond = {'pk_name': pk_name}
        fields = {'pk_val': 1}
        rec = MPkGenerator.tbl.find_one(query_cond, fields)
        pk_val = rec['pk_val']
        MPkGenerator.tbl.update_one(query_cond, {'$set', {'pk_val': pk_val+1}})
        return pk_val

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBrand.db = mongo_client['tcvdb']
        MBrand.tbl = MBrand.db['t_pk']