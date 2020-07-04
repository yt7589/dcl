#
import sys
from apps.wxs.wxs_dsm import WxsDsm

class WxsApp(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsApp'

    def startup(self, args):
        print('2020年7月无锡所招标应用')
        WxsDsm.initialize_db()