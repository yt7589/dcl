# 车标识别游戏
from apps.vbg.controller.c_vbg import CVbg
from apps.vbg.model.m_vehicle_brand import MVehicleBrand
from apps.vbg.model.m_file_odb import MFileOdb

class VbgApp(object):
    def __init__(self):
        self.name = 'apps.VbgApp'

    def startup(self):
        print('车标游戏')
        survey = CVbg.get_survey_data(50)
        print(survey)
        