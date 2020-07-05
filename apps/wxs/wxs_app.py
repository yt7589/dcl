#
import sys
from apps.wxs.wxs_dsm import WxsDsm
from apps.wxs.controller.c_brand import CBrand
from apps.wxs.controller.c_model import CModel

class WxsApp(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsApp'

    def startup(self, args):
        print('2020年7月无锡所招标应用')
        i_debug = 1
        if 1 == i_debug:
            self.exp()
            return
        WxsDsm.initialize_db()

    def exp(self):
        brand_vo = CBrand.get_brand_by_name('奔驰牌')
        print('品牌正例：{0}'.format(brand_vo))
        brand_vo = CBrand.get_brand_by_name('奔驰牌反例')
        print('品牌反例：{0}'.format(brand_vo))
        model_vo = CModel.get_model_by_name('奔驰_B级')
        print('车型正例：{0}'.format(model_vo))
        model_vo = CModel.get_model_by_name('奔驰_B级反例！')
        print('车型反例：{0}'.format(model_vo))