# 
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MDatasetSample(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MDatasetSample'

    @staticmethod
    def is_dataset_sample_exists(dataset_id, sample_id, sample_type):
        tbl = MMongoDb.db['t_dataset_sample']
        query_cond = {'dataset_id': dataset_id, 'sample_id': sample_id, 'sample_type': sample_type}
        fields = {'dataset_sample_id': 1}
        if tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_dataset_sample_by_infos(dataset_id, sample_id, sample_type):
        tbl = MMongoDb.db['t_dataset_sample']
        query_cond = {'dataset_id': dataset_id, 'sample_id': sample_id, 'sample_type': sample_type}
        fields = {'dataset_sample_id': 1, 'dataset_id': 1, 'sample_id': 1, 'sample_type': 1}
        return MMongoDb.convert_rec(tbl.find_one(query_cond, fields))

    @staticmethod
    def insert(dataset_sample_vo):
        '''
        向t_dataset_sample表中添加记录，dataset_sample_vo中包括：
            dataset_sample_id, dataset_id, sample_id, sample_type
        '''
        return MMongoDb.db['t_dataset_sample'].insert_one(dataset_sample_vo)