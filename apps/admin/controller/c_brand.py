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
    def get_known_brands(start_idx=1, amount=-1, sort_id=1, 
                sort_type=1):
        print('获取已有品牌列表...')
        #
        raw_set1 = set()
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        for brand_path in base_path.iterdir():
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            raw_set1.add(brand_name)
        print('raw_set1={0};'.format(len(raw_set1)))
        #
        raw_set2 = set()
        base_path1 = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/t1')
        for brand_path in base_path1.iterdir():
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            raw_set2.add(brand_name)
        print('raw_set2={0};'.format(len(raw_set2)))
        #
        novel_brands = set()
        for brand in raw_set2:
            if brand not in raw_set1:
                novel_brands.add(brand)
        print('novel brands: {0};'.format(len(novel_brands)))
        #
        print('拷贝新文件夹')
        src_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/t1'
        dst_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/t2'
        for brand in novel_brands:
            rst = os.popen('cp -R {0}/{1} {2}/.'.format(src_dir, brand, dst_dir))
            print('{0}: {1};'.format(brand, rst))
        brands = raw_set1
        for brand in raw_set2:
            brands.add(brand)
        print('brands={0};'.format(len(brands)))
        rst = {
            'total': len(brands),
            'brands': brands
        }
        return rst