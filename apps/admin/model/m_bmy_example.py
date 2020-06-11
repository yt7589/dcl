# 车辆品牌车型年款示例图片模型类
import pymongo
from apps.admin.model.m_mongodb import MMongoDb

class MBmyExample(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBmyExample'
        
    @staticmethod
    def get_bmy_example_vehicle_image_id(bmy_id):
        if MBmy.db is None:
            MBmy._initialize()
        query_cond = {'bmy_id': bmy_id}
        fields = {'vehicle_image_id': 1}
        return MMongoDb.convert_rec(MBmy.tbl.find_one(query_cond, fields))

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmy.db = mongo_client['tcvdb']
        MBmy.tbl = MBmy.db['t_bmy']

    