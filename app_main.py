#
import sys
import time
from pathlib import Path
import shutil
import utils.utils as du
from web.web_server import WebServer
import app_store
from utils.vao_test import VaoTest

MODE_TRAIN_WEB_SERVER = 101 # 运行训练阶段服务器
MODE_RUN_WEB_SERVER = 102 # 运行预测阶段服务器
MODE_TRAIN_MONITOR = 103 # 以Web服务器调整超参数
MODE_DRAW_ACCS_CURVE = 1001 # 绘制精度曲线
MODE_GET_BEST_CHPTS = 1002 # 在指定目录下获取最佳参数文件
MODE_CREATE_ST_CAR_DS = 1004 # 生成斯坦福汽车数据集
MODE_VAO_TEST = 1005 # 车管所测试工具

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

def main(args):
    print('细粒度图像识别系统')
    mode = MODE_VAO_TEST #MODE_TRAIN_MONITOR
    if MODE_DRAW_ACCS_CURVE == mode:
        du.draw_accs_curve()
    elif MODE_TRAIN_WEB_SERVER == mode:
        print('训练过程...')
    elif MODE_RUN_WEB_SERVER == mode:
        web_server = WebServer()
        web_server.setDaemon(True)
        web_server.start()
        web_server.join()
    elif MODE_TRAIN_MONITOR == mode:
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
    elif MODE_GET_BEST_CHPTS == mode:
        get_best_chpts()
    elif MODE_CREATE_ST_CAR_DS == mode:
        du.prepare_st_car_ds()
    elif MODE_VAO_TEST == mode:
        VaoTest.process_imported_vehicle()
    else:
        print('临时测试程序...')

if '__main__' == __name__:
    main({'__name__': __name__})