#
import pymongo

class MVehicleBrand(object):
    def __init__(self):
        self.name = 'apps.model.MVehicleBrand'
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.mongo_db = mongo_client['vehicle_brands']

    def insert_vehicle_brand(self, vehicle_brand_vo):
        rst = self.mongo_db.insert_one(vehicle_brand_vo)
        print('insert_id={0};'.format(rst.inserted_id))