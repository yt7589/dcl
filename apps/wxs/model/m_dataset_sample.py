# 
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MDatasetSample(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MDatasetSample'

    @staticmethod
    def is_dataset_sample_exists(dataset_id, sample_id, sample_type):
        if MDatasetSample.db is None:
            MDatasetSample._initialize()
        query_cond = {'dataset_id': dataset_id, 'sample_id': sample_id, 'sample_type': sample_type}
        fields = {'dataset_sample_id': 1}
        if MDatasetSample.tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_dataset_sample_by_infos(dataset_id, sample_id, sample_type):
        if MDatasetSample.db is None:
            MDatasetSample._initialize()
        query_cond = {'dataset_id': dataset_id, 'sample_id': sample_id, 'sample_type': sample_type}
        fields = {'dataset_sample_id': 1, 'dataset_id': 1, 'sample_id': 1, 'sample_type': 1}
        return MMongoDb.convert_rec(MDatasetSample.tbl.find_one(query_cond, fields))

    @staticmethod
    def insert(dataset_sample_vo):
        '''
        向t_dataset_sample表中添加记录，dataset_sample_vo中包括：
            dataset_sample_id, dataset_id, sample_id, sample_type
        '''
        if MDatasetSample.db is None:
            MDatasetSample._initialize()
        return MDatasetSample.tbl.insert_one(dataset_sample_vo)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDatasetSample.db = mongo_client['stpdb']
        MDatasetSample.tbl = MBmy.db['t_dataset_sample']