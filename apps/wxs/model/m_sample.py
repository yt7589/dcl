# 样本集模型类
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MSample(object):
    def __init__(self):
        self.name = 'apps.wxs.model.MSample'

    @staticmethod
    def is_sample_exists(img_file):
        query_cond = {'img_file': img_file}
        fields = {'sample_id': 1, 'bmy_id': 1}
        rec = MMongoDb.db['t_sample'].find_one(query_cond, fields)
        if rec is None:
            return False
        else:
            return True

    @staticmethod
    def get_sample_by_img_file(img_file):
        query_cond = {'img_file': img_file}
        fields = {'sample_id': 1, 'vin_id': 1, 'bmy_id': 1}
        return MMongoDb.db['t_sample'].find_one(query_cond, fields)

    @staticmethod
    def insert(sample_vo):
        '''
        向t_sample表中添加记录，sample_vo中包括：
            sample_id, vin_id, bmy_id, img_file
        '''
        return MMongoDb.db['t_sample'].insert_one(sample_vo)

    @staticmethod
    def get_vin_samples(vin_id):
        query_cond = {'vin_id': vin_id}
        fields = {"img_file": 1, "bmy_id": 1}
        return MMongoDb.convert_recs(MMongoDb.db['t_sample'].find(query_cond, fields).sort([('img_file', 1)]))
