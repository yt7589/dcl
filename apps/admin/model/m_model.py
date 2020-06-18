# 车型模型类
import pymongo

class MModel(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MModel'

    @staticmethod
    def get_model_by_name(model_name):
        if MModel.db is None:
            MModel._initialize()
        query_cond = {'model_name': model_name}
        fields = {'model_id': 1, 'model_name': 1, 'model_num': 1}
        return MModel.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(model_vo):
        if MModel.db is None:
            MModel._initialize()
        return MModel.tbl.insert_one(model_vo)

    @staticmethod
    def process_brand_rename(old_brand_name, new_brand_name):
        if MModel.db is None:
            MModel._initialize()
        query_cond = {'model_name': {'$regex': '^{0}'.format(old_brand_name)}}
        fields = {'model_id': 1, 'model_name': 1}
        recs = MModel.tbl.find(query_cond, fields)
        for rec in recs:
            print('### {0};'.format(rec))
            MModel._update_model_name(rec['model_id'], rec['model_name'], new_brand_name)
        #new_values = {'$set': {'model_name': modelName}}

    @staticmethod
    def _update_model_name(model_id, raw_name, bn):
        if MModel.db is None:
            MModel._initialize()
        query_cond = {'model_id': model_id}
        arrs0 = raw_name.split('_')
        model_name = arrs0[1]
        bm_name = '{0}_{1}'.format(bn, model_name)
        new_values = {'$set': {'model_name': bm_name}}
        print('query_cond: {0}; new_values: {1};'.format(query_cond, new_values))
        MModel.tbl.update_one(query_cond, new_values)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MModel.db = mongo_client['tcvdb']
        MModel.tbl = MModel.db['t_model']
