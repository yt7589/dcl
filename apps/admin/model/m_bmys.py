# 处理bmy表

class MBmys(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MBmys'

    @staticmethod
    def insert(bmy_vo):
        if MBmys.tbl is None:
            MBmys._initialize()
        MBmys.tbl.insert_one(bmy_vo)

    @staticmethod
    def delete_all():
        if MBmys.tbl is None:
            MBmys._initialize()
        MBmys.tbl.delete_many({})
    
    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MBrand.db = mongo_client['tcvdb']
        MBrand.tbl = MBrand.db['bmys']