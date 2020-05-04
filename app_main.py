#
from pathlib import Path
import utils.utils as du

MODE_TRAIN_WEB_SERVER = 101 # 运行训练阶段服务器
MODE_RUN_WEB_SERVER = 102 # 运行预测阶段服务器
MODE_DRAW_ACCS_CURVE = 1001 # 绘制精度曲线
MODE_GET_BEST_CHPTS = 1002 # 在指定目录下获取最佳参数文件

def get_best_chpts():
    chpts_dir = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/fgvc/dcl/net_model/training_descibe_5216_CUB/')
    files = [x for x in chpts_dir.iterdir() if chpts_dir.is_dir()]
    max_acc1 = -1.0
    max_chpt = ''
    for fi in files:
        arrs = str(fi).split('_')
        acc1 = float(arrs[3])
        if acc1 > max_acc1:
            max_acc1 = acc1
            max_chpt = fi
        print('acc1={0};'.format(max_acc1))
    print('max: {0};'.format(max_chpt))


def main(args):
    print('细粒度图像识别系统')
    mode = MODE_GET_BEST_CHPTS
    if MODE_DRAW_ACCS_CURVE == mode:
        du.draw_accs_curve()
    elif MODE_TRAIN_WEB_SERVER == mode:
        print('训练过程...')
    elif MODE_RUN_WEB_SERVER == mode:
        print('预测过程...')
    elif MODE_GET_BEST_CHPTS == mode:
        get_best_chpts()
    else:
        print('临时测试程序...')

if '__main__' == __name__:
    main({})