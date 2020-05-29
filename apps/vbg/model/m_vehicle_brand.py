#
import pymongo

class MVehicleBrand(object):
    def __init__(self):
        self.name = 'apps.model.MVehicleBrand'
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.mongo_db = mongo_client['tcvdb']
        self.tbl = self.mongo_db['vehicle_brands']

    def insert_vehicle_brand(self, vehicle_brand_vo):
        rst = self.tbl.insert_one(vehicle_brand_vo)
        print('insert_id={0};'.format(rst.inserted_id))

    def update_by_vehicle_brand_id(self, vehicle_brand_id, vehicle_brand_vo):
        query_cond = {'vehicle_brand_id': vehicle_brand_id}
        new_values = { '$set': { 
            'place_of_origin':  vehicle_brand_vo['place_of_origin']
        } }
        self.tbl.update_one(query_cond, new_values)

    def delete_by_vehicle_brand_id(self, vehicle_brand_id):
        query_cond = {'vehicle_brand_id': vehicle_brand_id}
        self.tbl.delete_one(query_cond)

    def query_by_vehicle_brand_name(self, vehicle_brand_name):
        query_cond = {'vehicle_brand_name': vehicle_brand_name}
        fields = {'vehicle_brand_name': 1, 'vehicle_brand_alias': 1}
        return self.tbl.find(query_cond, fields)