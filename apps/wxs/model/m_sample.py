# 样本集模型类
import pymongo
from apps.wxs.model.m_mongodb import MMongoDb

class MSample(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.wxs.model.MSample'

    @staticmethod
    def is_sample_exists(img_file):
        if MSample.db is None:
            MSample._initialize()
        query_cond = {'img_file': img_file}
        fields = {'sample_id': 1, 'bmy_id': 1}
        if MSample.tbl.find_one(query_cond, fields) is None:
            return False
        else:
            return True

    @staticmethod
    def get_sample_by_img_file(img_file):
        if MSample.db is None:
            MSample._initialize()
        query_cond = {'img_file': img_file}
        fields = {'sample_id': 1, 'vin_id': 1, 'bmy_id': 1}
        return MSample.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(sample_vo):
        '''
        向t_sample表中添加记录，sample_vo中包括：
            sample_id, vin_id, bmy_id, img_file
        '''
        if MSample.db is None:
            MSample._initialize()
        return MSample.tbl.insert_one(sample_vo)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MVin.db = mongo_client['stpdb']
        MVin.tbl = MVin.db['t_sample']
