#
import sys
import time
from pathlib import Path
import shutil
#import utils.utils as du
#from web.web_server import WebServer
import app_store
from utils.vao_test import VaoTest
from utils.data_preprocessor import DataPreprocessor
from utils.ds_manager import DsManager
from apps.cluster.cluster_app import ClusterApp
from apps.vbg.vbg_app import VbgApp
from apps.admin.admin_app import AdminApp
from apps.admin.controller.c_ggh_bmy import CGghBmy
from apps.admin.controller.c_brand import CBrand
from apps.admin.controller.c_ggh_bmy import CGghBmy
from apps.admin.controller.c_data_source import CDataSource
from apps.admin.controller.c_bmy import CBmy
from apps.admin.controller.c_vehicle_image import CVehicleImage
from apps.admin.controller.c_delta_ds import CDeltaDs
from apps.wxs.wxs_app import WxsApp
from apps.wxs.model.m_mongodb import MMongoDb
import pymongo
from apps.siamese.siamese_app import SiameseApp
from utils.onnx_exporter import OnnxExporter

MODE_TRAIN_WEB_SERVER = 101 # 运行训练阶段服务器
MODE_RUN_WEB_SERVER = 102 # 运行预测阶段服务器
MODE_TRAIN_MONITOR = 103 # 以Web服务器调整超参数
MODE_DRAW_ACCS_CURVE = 1001 # 绘制精度曲线
MODE_GET_BEST_CHPTS = 1002 # 在指定目录下获取最佳参数文件
MODE_CREATE_ST_CAR_DS = 1004 # 生成斯坦福汽车数据集
MODE_VAO_TEST = 1005 # 车管所测试工具
MODE_DATA_PREPROCESSOR = 1006 # 数据预处理器
MODE_DS_MANAGER = 1007 # 数据集管理器程序
MODE_CLUSTER_IMAGE = 1008 # 探索使用图像聚类方法
MODE_LOCAL_STANFORD_CARS = 1009 # 本地运行Stanford Cars数据集
MODE_TEST_MONGODB = 1010 # MONGODB学习
MODE_TEST_ADMIN = 1011 # 测试后台管理功能
MODE_TEST_WEB_API = 1012 # 测试Web接口

def get_best_chpts():
    chpts_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/net_model/training_descibe_5412_CUB/')
    store_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/chpts/senet154/')
    files = [x for x in chpts_dir.iterdir() if chpts_dir.is_dir()]
    max_acc1 = -1.0
    max_chpt = ''
    max_file = ''
    for fi in files:
        arrs = str(fi).split('_')
        arrs1 = str(fi).split('/')
        acc1 = float(arrs[-2])
        if acc1 > max_acc1:
            max_acc1 = acc1
            max_chpt = fi
            max_file = arrs1[-1]
    print('max: {0}; = {1};'.format(max_chpt, max_file))
    dst_file = '{0}/{1}'.format(store_dir, max_file)
    print('dst_file: {0};'.format(dst_file))
    shutil.copy(max_chpt, dst_file)

def temp_func():
    VaoTest.draw_b86_train_curve()

def test_web_api():
    worker_id = '102'
    recs = CDeltaDs.get_check_delta_ds_detls(worker_id)
    for rec in recs:
        print('#: {0};'.format(rec))

def main(args):
    MMongoDb._initialize()
    ii = 1
    if 1 == ii:
        #rst = MMongoDb.db.t_sample.create_index([('vin_id', pymongo.ASCENDING), ('img_file', pymongo.ASCENDING)], unique=True)
        #print('index rst: {0};'.format(rst))
        #onnx_exporter = OnnxExporter()
        #onnx_exporter.run_onnx()
        app = WxsApp()
        #app = SiameseApp()
        app.startup(args)
        return
    print('细粒度图像识别系统')
    mode = MODE_DS_MANAGER #MODE_TRAIN_MONITOR
    if MODE_DRAW_ACCS_CURVE == mode:
        #du.draw_accs_curve()
        pass
    elif MODE_TRAIN_WEB_SERVER == mode:
        print('训练过程...')
    elif MODE_RUN_WEB_SERVER == mode:
        '''
        web_server = WebServer()
        web_server.setDaemon(True)
        web_server.start()
        web_server.join()
        '''
        pass
    elif MODE_TRAIN_MONITOR == mode:
        '''
        web_server = WebServer()
        web_server.setDaemon(True)
        web_server.start()
        # start training process
        for i in range(100):
            try:
                time.sleep(3)
            except:
                print('exceptions... stop thread')
                break
            print('loop_{0}: lr={1};'.format(i, app_store.vars['lr']))
        '''
        pass
    elif MODE_GET_BEST_CHPTS == mode:
        get_best_chpts()
    elif MODE_CREATE_ST_CAR_DS == mode:
        #du.prepare_st_car_ds()
        pass
    elif MODE_VAO_TEST == mode:
        VaoTest.startup()
    elif MODE_DATA_PREPROCESSOR == mode:
        DataPreprocessor.startup()
    elif MODE_DS_MANAGER == mode:
        DsManager.startup()
    elif MODE_CLUSTER_IMAGE == mode:
        app = ClusterApp()
        app.startup()
    elif MODE_LOCAL_STANFORD_CARS == mode:
        DataPreprocessor.startup()
    elif MODE_TEST_MONGODB == mode:
        app = VbgApp()
        app.startup()
    elif MODE_TEST_ADMIN == mode:
        AdminApp.startup()
    elif MODE_TEST_WEB_API == mode:
        test_web_api()
    else:
        print('临时测试程序...')
        temp_func()

if '__main__' == __name__:
    main({'__name__': __name__})
