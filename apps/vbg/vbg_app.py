# 车标识别游戏
from apps.vbg.model.m_vehicle_brand import MVehicleBrand

class VbgApp(object):
    def __init__(self):
        self.name = 'apps.VbgApp'

    def startup(self):
        print('车标游戏')
        vehicle_brand_id = 1
        model = MVehicleBrand()
        '''
        vehicle_brand_vo = {
            'place_of_origin': '韩国.汉城'
        }
        '''
        model.delete_by_vehicle_brand_id(
            vehicle_brand_id
        )