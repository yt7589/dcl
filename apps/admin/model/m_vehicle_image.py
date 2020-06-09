# 车辆图片文件模型类
import pymongo

class MVehicleImage(object):
    db = None
    tbl = None

    def __init__(self):
        self.name = 'apps.admin.model.MVehicleImage'

    @staticmethod
    def get_vehicle_image_vo(filename):
        if MVehicleImage.db is None:
            MVehicleImage._initialize()
        query_cond = {'filename': filename}
        fields = {'vehicle_image_id': 1, 'filename': 1, 'full_path': 1}
        return MVehicleImage.tbl.find_one(query_cond, fields)

    @staticmethod
    def insert(vehicle_image_vo):
        if MVehicleImage.db is None:
            MVehicleImage._initialize()
        return MVehicleImage.tbl.insert_one(vehicle_image_vo)

    @staticmethod
    def get_vehicle_image_full_path(vehicle_image_id):
        '''
        获取vehicle_image_id对应图片的全路径文件名
        '''
        if MVehicleImage.db is None:
            MVehicleImage._initialize()
        query_cond = {'vehicle_image_id': vehicle_image_id}
        fields = {'filename': 1, 'full_path': 1}
        return MVehicleImage.tbl.find_one(query_cond, fields)

    @staticmethod
    def _initialize():
        mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
        MVehicleImage.db = mongo_client['tcvdb']
        MVehicleImage.tbl = MVehicleImage.db['t_vehicle_image']
