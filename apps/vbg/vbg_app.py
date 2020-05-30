# 车标识别游戏
from apps.vbg.model.m_vehicle_brand import MVehicleBrand
from apps.vbg.model.m_file_odb import MFileOdb

class VbgApp(object):
    def __init__(self):
        self.name = 'apps.VbgApp'

    def startup(self):
        print('车标游戏')
        model = MVehicleBrand()
        total = model.get_total_recs()
        print('count={0};'.format(total))
        rec = model.get_random_rec(total)
        print('^_^ rec: {0};'.format(rec))