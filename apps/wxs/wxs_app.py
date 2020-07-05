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
        i_debug = 10
        if 1 == i_debug:
            self.exp()
            return
        WxsDsm.initialize_db()

    def exp(self):
        ds_file = './datasets/CUB_200_2011/anno/train_ds_v4.txt'
        train_id = self._t1(ds_file)
        ds_file = './datasets/CUB_200_2011/anno/test_ds_v4.txt'
        test_id = self._t1(ds_file)
        print('{0} vs {1};'.format(train_id, test_id))

    def _t1(self, ds_file):
        max_fgvc_id = 0
        with open(ds_file, 'r', encoding='utf-8') as nfd:
            for line in nfd:
                row = line.strip()
                arrs0 = row.split('*')
                fgvc_id = int(arrs0[1])
                if fgvc_id > max_fgvc_id:
                    max_fgvc_id = fgvc_id
        return max_fgvc_id