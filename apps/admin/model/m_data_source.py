# 数据源模型类，用于管理训练和测试样本，生成训练和测试数据集
import pymongo
from apps.admin.model.m_mongodb import MMongoDb
from apps.admin.model.m_bmy import MBmy

class MDataSource(object):
    db = None
    tbl = None
    #
    SAMPLE_TYPE_TRAIN = 1
    SAMPLE_TYPE_VALIDATE = 2
    SAMPLE_TYPE_TEST = 3

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
        sample_type = MDataSource.SAMPLE_TYPE_TRAIN # 训练数据集
        return MDataSource.get_raw_bmy_samples(bmy_id, sample_type)

    @staticmethod
    def get_bmy_test_samples(bmy_id):
        sample_type= MDataSource.SAMPLE_TYPE_TEST # 
        return MDataSource.get_raw_bmy_samples(bmy_id, sample_type)

    @staticmethod
    def get_raw_bmy_samples(bmy_id, sample_type):
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'bmy_id': bmy_id, 'type': sample_type, 
                    'state': 0}
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
    def get_train_data_sources():
        if MDataSource.db is None:
            MDataSource._initialize()
        all_samples = []
        recs = MBmy.get_bmy_ids()
        for rec in recs:
            bmy_id = rec['bmy_id']
            train_samples = MDataSource.get_bmy_samples(bmy_id, MDataSource.SAMPLE_TYPE_TRAIN)
            test_samples = []
            if len(train_samples) < 10:
                # 取出Test中的样本
                test_samples = MDataSource.get_bmy_samples(bmy_id, MDataSource.SAMPLE_TYPE_TEST)
            all_samples = all_samples + train_samples + test_samples
        return all_samples
        

    @staticmethod
    def get_bmy_samples(bmy_id, sample_type):
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'data_source_id': {'$gt': 0}, 
            'state': 1, 'type': sample_type,
            'bmy_id': bmy_id
        }
        fields = {'bmy_id': 1, 'vehicle_image_id': 1}
        return MMongoDb.convert_recs(MDataSource.tbl.find(query_cond, fields))

    @staticmethod
    def get_test_data_sources():
        if MDataSource.db is None:
            MDataSource._initialize()
        query_cond = {'data_source_id': {'$gt': 0}, 'state': 1, 'type': MDataSource.SAMPLE_TYPE_TEST}
        fields = {'bmy_id': 1, 'vehicle_image_id': 1}
        return MMongoDb.convert_recs(MDataSource.tbl.find(query_cond, fields))

    @staticmethod
    def get_bmy_current_vehicle_image_id(bmy_id, prev_vehicle_image_id, mode):
        if MDataSource.db is None:
            MDataSource._initialize()
        if mode == 1:
            query_cond = {'bmy_id': bmy_id, 'vehicle_image_id': {'$gt': prev_vehicle_image_id}}
        else:
            query_cond = {'bmy_id': bmy_id, 'vehicle_image_id': {'$lt': prev_vehicle_image_id}}
        fields = {'vehicle_image_id': 1}
        recs = MMongoDb.convert_recs(MDataSource.tbl.find(query_cond, fields).sort('vehicle_image_id', 1))
        print('######### recs: {0};'.format(recs))
        if recs is not None:
            return recs[0]
        else:
            return {'vehicle_image_id': 0}


    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDataSource.db = mongo_client['tcvdb']
        MDataSource.tbl = MDataSource.db['t_data_source']

