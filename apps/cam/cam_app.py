#
from apps.wxs.wxs_app import WxsApp

class CamApp(object):

    def __init__(self):
        self.refl = 'apps.cam.CamApp'

    def startup(self, args):
        i_debug = 10
        if 1 == i_debug:
            app = WxsApp()
            app.startup(args)
            return
        print('模型热力图绘制应用 v0.0.1')