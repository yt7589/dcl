#
import pymongo

class MVehicleBrand(object):
    def __init__(self):
        self.name = 'apps.model.MVehicleBrand'
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.mongo_db = mongo_client['tcvdb']

    def insert_vehicle_brand(self, vehicle_brand_vo):
        tbl = self.mongo_db['vehicle_brands']
        rst = tbl.insert_one(vehicle_brand_vo)
        print('insert_id={0};'.format(rst.inserted_id))