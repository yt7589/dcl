# 车标识别游戏
from apps.vbg.model.m_vehicle_brand import MVehicleBrand

class VbgApp(object):
    def __init__(self):
        self.name = 'apps.VbgApp'

    def startup(self):
        print('车标游戏')
        model = MVehicleBrand()
        vehicle_brand_vo = {
            'vehicle_brand_id': 2,
            'vehicle_brand_name': '雪铁龙',
            'vehicle_brand_alias': 'CITROEN',
            'place_of_origin': '法国·巴黎'
        }
        model.insert_vehicle_brand(vehicle_brand_vo)