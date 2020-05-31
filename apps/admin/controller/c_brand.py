# 品牌控制器类
import os
from pathlib import Path

class CBrand(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBrand'
    
    @staticmethod
    def get_known_brands_api(start_idx=1, amount=-1, sort_id=1,
                 sort_type=1):
        '''
        获取已有品牌列表，包括品牌名称，年款数，图像数，
        支持按品牌名称、年款数、图像数排序，返回JSON响应
        '''
        brands = CBrand.get_known_brands(start_idx, amount, 
                    sort_id, sort_type)
        data = {
            'brands': brands
        }
        return data

    @staticmethod
    def sort_by_num(item):
        return item['num']
    
    @staticmethod
    def get_known_brands(start_idx=1, amount=-1, sort_id=1, 
                sort_type=1):
        print('获取已有品牌列表...')
        #
        brands = set()
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        for brand_path in base_path.iterdir():
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            brands.add(brand_name)
        print('v1 brands={0};'.format(len(brands)))
        bns = []
        bns.append({'name': 'b1', 'num': 10})
        bns.append({'name': 'b2', 'num': 5})
        bns.append({'name': 'b3', 'num': 18})
        print(bns)
        b1 = sorted(bns, key=CBrand.sort_by_num, reverse=True)
        print('b1: {0};'.format(b1))
        rst = {
            'total': len(brands),
            'brands': brands
        }
        return rst