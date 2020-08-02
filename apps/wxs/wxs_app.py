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
        '''
        利用所里最新Excel表格内容为主，work/ggh2_to_bmy_dict.txt内容为辅，
        生成数据库中t_brand、t_model、t_bmy、t_vin表格中内容
        '''
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
        '''
        生成Cnstream要求的车辆品牌车型年款标签文件，格式为：
        {"品牌编号", "车型编号", "年款编号", "品牌_车型_年款"},{...},
        {...}
        '''
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
        '''
        处理所里测试集中5664张正确图片中新车型和新年款记录，添加到数据集中
        '''
        #WxsDsm.process_unknown_wxs_tds()
        '''
        根据错误分类样本列表，形成便于人工浏览的网页，保存于../../w1/es目录下
        '''
        #WxsDsm.generate_error_samples_html()
        #WxsDsm.fix_test_ds_brand_errors()
        '''
        获取所里最新Excel表格中的品牌车型年款
        '''
        #WxsDsm.get_wxs_bmys()
        '''
        从t_vin表中获取当前不在所里5731个品牌车型年款中的车辆识别码
        '''
        #WxsDsm.get_non_wxs_vins()
        '''
        将无锡所测试文件放到Excel表格中
        '''
        #WxsDsm.generate_wxs_tds_table()
        '''
        求出无锡所Excel表格中每个车辆识别码的图片数，并列出图片数为零的
        车辆识别码编号
        '''
        #WxsDsm.get_wxs_vin_id_img_num()
        '''
        获取无锡所品牌列表
        '''
        #WxsDsm.get_wxs_brands()
        '''
        从samples.txt文件中统计出品牌数、车型数、年款数
        '''
        #WxsDsm.get_brand_bm_bmy_of_samples()
        '''
        实现先预测出品牌类别，然后从年款头中除该品牌对应的年款索引外的其他
        类别全部清零，将年款头的内容输出作为输出
        '''
        #WxsDsm.bind_brand_head_bmy_head()
        '''
        求出无锡所测试集品牌与当前涉及的171个品牌的不同
        '''
        #WxsDsm.get_diff_wxs_tds_brands()
        '''
        标出无锡所测试集中可能出错的图片
        '''
        #WxsDsm.mark_error_img_in_wxs_tds()
        WxsDsm.exp001()

        

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