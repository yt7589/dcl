# 各种工具类的实现
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class WxsUtils(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsUtils'

    @staticmethod
    def draw_tds_acc_curve():
        base_path = Path('/media/zjkj/work/yantao/fgvc/dcl/net_model/training_descibe_8223_CUB')
        x = []
        y = []
        for file_obj in base_path.iterdir():
            full_path = str(file_obj)
            arrs0 = full_path.split('/')
            log_file = arrs0[-1]
            arrs1 = log_file.split('_')
            x.append(int(arrs1[2]))
            y.append(float(arrs1[3]))
        # 绘制测试集上精度变化曲线
        x = np.array(x)
        y = np.array(y)
        fig, ax = plt.subplots()
        ax.plot(x, y, label='accuracy')
        ax.set_xlabel('steps')
        ax.set_ylabel('accuracy')
        ax.set_title("accuracy curve")
        ax.legend()
        plt.show()