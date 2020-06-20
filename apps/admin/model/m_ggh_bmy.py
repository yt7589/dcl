# 公告号与品牌车型年款对应关系
import pymongo

class MGghBmy(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MGghBmy'

    @staticmethod
    def get_ggh_bmy_by_code(ggh_code):
        if MGghBmy.db is None:
            MGghBmy._initialize()
        query_cond = {'ggh_code': ggh_code}
        fields = {'ggh_bmy_id': 1, 'ggh_code': 1, 'bmy_id': 1}
        return MGghBmy.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(ggh_bmy_vo):
        if MGghBmy.db is None:
            MGghBmy._initialize()
        return MGghBmy.tbl.insert_one(ggh_bmy_vo)

    @staticmethod
    def is_ggh_exists(ggh_code):
        '''
        公告号本身存在，或者以其为前缀的公告号存在
        '''
        if MGghBmy.db is None:
            MGghBmy._initialize()
        query_cond = {'ggh_code': {'$regex': '^{0}'.format(ggh_code)}}
        fields = {'ggh_bmy_id': 1, 'bmy_id': 1}
        cnt = MGghBmy.tbl.find(query_cond, fields).count()
        if cnt > 0:
            return True
        else:
            return False


    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MGghBmy.db = mongo_client['tcvdb']
        MGghBmy.tbl = MGghBmy.db['t_ggh_bmy']