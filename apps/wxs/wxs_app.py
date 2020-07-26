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
        #WxsDsm.exp001()
        #WxsDsm.initialize_db()
        ''' 
        从fgvc_dataset/raw和guochanchezuowan_all目录生成样本列表
        '''
        #WxsDsm.generate_samples()
        '''
        生成原始数据集，采用稀疏品牌车型年款编号
        '''
        #WxsDsm.generate_dataset()
        '''
        将品牌车型年款变为0开始递增的序号
        '''
        #WxsDsm.get_simplified_bmys()
        '''
        向数据集中加入品牌信息
        '''
        #WxsDsm.convert_to_brand_ds_main()
        '''
        找出损坏的图片文件
        '''
        #WxsDsm.find_bad_images()
        #WxsDsm.report_current_status()
        #WxsDsm.exp001()
        #WxsDsm.get_fine_wxs_dataset()
        #WxsDsm.generate_wxs_bmy_csv()
        #WxsDsm.generate_zjkj_cambricon_labels()
        ''' 根据正确的测试集图片文件名，查出当前的品牌车型年款编号，没有的用-1表示，形成CSV文件 '''
        #WxsDsm.generate_test_ds_bmy_csv()
        ''' 生成Pipeline测试评价数据，将测试集中的图片文件拷贝到指定目录下 '''
        #WxsDsm.copy_test_ds_images_for_cnstream()
        '''
        集成无锡所测试集数据
        '''
        #WxsDsm.integrate_wxs_test_ds()
        '''
        生成车辆识别码和品牌车型年款对应关系表，用于修正所里品牌车型年款不合理的地方。2020.07.25
        '''
        #WxsDsm.generate_vin_bmy_csv()

        

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