# 数据源模型类，用于管理训练和测试样本，生成训练和测试数据集
import pymongo

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
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MDataSource.db = mongo_client['tcvdb']
        MDataSource.tbl = MDataSource.db['t_data_source']

