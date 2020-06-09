# 品牌车型年款模型类
import pymongo

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
        return MBmy.tbl.find({}, fields)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBmy.db = mongo_client['tcvdb']
        MBmy.tbl = MBmy.db['t_bmy']