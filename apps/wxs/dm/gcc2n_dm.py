# 国产车2n目录相关统计信息
from pathlib import Path

class Gcc2nDm(object):
    def __init__(self):
        self.refl = 'apps.wxs.dm.Gcc2nDm'
        
    @staticmethod
    def get_gcc2n_vcs():
        '''
        获取guochanche_2n目录下所有车辆识别码
        '''
        gcc2_vcs = set()
        base_path = Path('/media/zjkj/work/guochanche_2n')
        for vco in base_path.iterdir():
            vc = str(vco)
            gcc2_vcs.add(vc)
        base_path = Path('/media/zjkj/work/g2ne')
        for sf1 in base_path.iterdir():
            for sf2 in sf1.iterdir():
                vc = str(sf2)
                gcc2_vcs.add(vc)
        return gcc2_vcs