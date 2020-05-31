# 用于调试后台管理程序的辅助程序
from apps.admin.controller.c_brand import CBrand

class AdminApp(object):
    def __init__(self):
        self.name = 'apps.admin.AdminApp'

    @staticmethod
    def startup():
        print('后台管理程序调试工具类 v0.0.1')
        rst = CBrand.get_known_brands(start_idx=1, amount=-1, sort_id=1, 
                    sort_type=1)
        print('已有品牌数量：{0};'.format(len(rst)))
        for brand_obj in rst:
            print('{0}: {1};'.format(brand_obj['name'], brand_obj['num']))