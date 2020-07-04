#
import sys
from apps.wxs.wxs_dsm import WxsDsm
from apps.wxs.controller.c_brand import CBrand

class WxsApp(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsApp'

    def startup(self, args):
        print('2020年7月无锡所招标应用')
        i_debug = 10
        if 1 == i_debug:
            self.exp()
            return
        WxsDsm.initialize_db()

    def exp(self):
        brand_vo = CBrand.get_brand_by_name('奔驰牌')
        print(brand_vo)