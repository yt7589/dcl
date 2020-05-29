# 车标识别游戏
from apps.vbg.model.m_vehicle_brand import MVehicleBrand

class VbgApp(object):
    def __init__(self):
        self.name = 'apps.VbgApp'

    def startup(self):
        print('车标游戏')
        vehicle_brand_name = '雪铁龙'
        model = MVehicleBrand()
        recs = model.query_by_vehicle_brand_name(vehicle_brand_name)
        print('recs: {0};'.format(type(recs)))
        for rec in recs:
            print(rec)