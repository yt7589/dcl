# 品牌车型年款模型类
import pymongo
from apps.admin.model.m_mongodb import MMongoDb

class MBmy(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBmy'

    @staticmethod
    def get_bmy_by_name(bmy_name):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_name': bmy_name}
        fields = {'bmy_id': 1, 'bmy_name': 1, 'bmy_num': 1}
        return MBmy.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(bmy_vo):
        if MBmy.db is None:
            MBmy._initialize()
        return MBmy.tbl.insert_one(bmy_vo)

    @staticmethod
    def get_bmy_ids():
        if MBmy.db is None:
            MBmy._initialize()
        fields = {'bmy_id': 1}
        return MMongoDb.convert_recs(MBmy.tbl.find({}, fields).sort('bmy_id', 1))

    @staticmethod
    def get_bmy_name_by_id(bmy_id):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {"bmy_id": bmy_id}
        fields = {"bmy_name": 1}
        return MMongoDb.convert_rec(MBmy.tbl.find_one(query_cond, fields))

    @staticmethod
    def get_bmys():
        '''
        获取品牌车型年款列表，返回值：[{'bmy_id': 1, 'bmy_name': '奔驰_S级_2012'}]
        '''
        if MBmy.db is None:
            MBmy._initialize()
        fields = {'bmy_id': 1, 'bmy_name': 1}
        return MMongoDb.convert_recs(MBmy.tbl.find({}, fields).sort('bmy_id', 1))


        

    @staticmethod
    def process_brand_rename(old_brand_name, new_brand_name):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_name': {'$regex': '^{0}'.format(old_brand_name)}}
        fields = {'bmy_id': 1, 'bmy_name': 1}
        recs = MBmy.tbl.find(query_cond, fields)
        for rec in recs:
            print('### {0};'.format(rec))
            MBmy._update_bmy_name(rec['bmy_id'], rec['bmy_name'], new_brand_name)
        #new_values = {'$set': {'model_name': modelName}}

    @staticmethod
    def _update_bmy_name(bmy_id, raw_name, bn):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_id': bmy_id}
        arrs0 = raw_name.split('_')
        model_name = arrs0[1]
        year_name = arrs0[2]
        bmy_name = '{0}_{1}_{2}'.format(bn, model_name, year_name)
        new_values = {'$set': {'bmy_name': bmy_name}}
        print('query_cond: {0}; new_values: {1};'.format(query_cond, new_values))
        MBmy.tbl.update_one(query_cond, new_values)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmy.db = mongo_client['tcvdb']
        MBmy.tbl = MBmy.db['t_bmy']