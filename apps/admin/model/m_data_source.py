# 数据源模型类，用于管理训练和测试样本，生成训练和测试数据集
import pymongo
from apps.admin.model.m_mongodb import MMongoDb

class MDataSource(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MDataSource'

    @staticmethod
    def get_data_source_by_vid(vehicle_image_id, bmy_id):
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'vehicle_image_id': vehicle_image_id, 'bmy_id': bmy_id}
        fields = {'data_source_id': 1, 'bmy_id': 1, 'state': 1, 'type': 1}
        return MDataSource.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(data_source_vo):
        if MDataSource.db is None:
            MDataSource._initialize()
        return MDataSource.tbl.insert_one(data_source_vo)

    @staticmethod
    def get_bmy_raw_train_samples(bmy_id):
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'bmy_id': bmy_id}
        fields = {'data_source_id': 1, 'vehicle_image_id': 1}
        return MMongoDb.convert_recs(MDataSource.tbl.find(query_cond, fields))

    @staticmethod
    def update_state(data_source_id, state):
        '''
        更新对应记录的state
        '''
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'data_source_id': data_source_id}
        new_values = {"$set": {"state": state}}
        MDataSource.tbl.update_one(query_cond, new_values)

    @staticmethod
    def get_all_data_sources():
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'data_source_id': {'$gt': 0}, 'state': 1}
        fields = {'bmy_id': 1, 'vehicle_image_id': 1}
        return MMongoDb.convert_recs(MDataSource.tbl.find(query_cond, fields))


    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDataSource.db = mongo_client['tcvdb']
        MDataSource.tbl = MDataSource.db['t_data_source']

