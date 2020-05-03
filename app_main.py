#
import utils.utils as du

MODE_TRAIN_WEB_SERVER = 101 # 运行训练阶段服务器
MODE_RUN_WEB_SERVER = 102 # 运行预测阶段服务器
MODE_DRAW_ACCS_CURVE = 1001 # 绘制精度曲线

def main(args):
    print('细粒度图像识别系统')
    mode = MODE_DRAW_ACCS_CURVE
    if MODE_DRAW_ACCS_CURVE == mode:
        du.draw_accs_curve()
    elif MODE_TRAIN_WEB_SERVER == mode:
        print('训练过程...')
    elif MODE_RUN_WEB_SERVER == mode:
        print('预测过程...')
    else:
        print('临时测试程序...')

if '__main__' == __name__:
    main({})