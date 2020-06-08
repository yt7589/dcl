# 品牌控制器类
import os
from pathlib import Path
from apps.admin.model.m_brand import MBrand
from apps.admin.model.m_pk_generator import MPkGenerator

class CBrand(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CBrand'

    @staticmethod
    def add_brand(brand_name):
        brand_vo = MBrand.get_brand_by_name(brand_name)
        if brand_vo is None:
            brand_id = MPkGenerator.get_pk('brand')
            brand_vo = {
                'brand_id': brand_id,
                'brand_name': brand_name,
                'brand_pics': 0
            }
            MBrand.insert(brand_vo)
        return brand_vo['brand_id']

    @staticmethod
    def get_unknown_brands():
        must_brands = CBrand.get_must_brands()
        unknown_brands = []
        for brand in must_brands:
            print('### {0};'.format(brand))
            if MBrand.get_brand_by_name(brand) is None:
                unknown_brands.append(brand)
        return unknown_brands

    @staticmethod
    def get_must_brands():
        must_brands = []
        with open('./datasets/bno_bn.txt', 'r', encoding='utf-8') as fd:
            for line in fd:
                arrs0 = line.split(':')
                bno = arrs0[0]
                brand_name = arrs0[1][:-1]
                must_brands.append(brand_name)
        return must_brands

    







    
    @staticmethod
    def get_known_brands_api(start_idx=1, amount=-1, sort_id=1,
                 sort_type=1):
        '''
        从mongodb中获取已有品牌列表，包括品牌名称，年款数，图像数，
        支持按品牌名称、年款数、图像数排序，返回JSON响应
        '''
        recs = MBrand.query_brands()
        brands = []
        for rec in recs:
            brand = {
                'brand_id': rec['brand_id'],
                'brand_name': rec['brand_name'],
                'brand_num': rec['brand_num']
            }
            brands.append(brand)
        data = {
            'total': len(brands),
            'brands': brands
        }
        return data

    @staticmethod
    def get_refresh_known_brands_api(start_idx=1, amount=-1, sort_id=1,
                 sort_type=1):
        '''
        获取已有品牌列表，包括品牌名称，年款数，图像数，
        支持按品牌名称、年款数、图像数排序，返回JSON响应，从目录中统计实时数据，并
        将结果保存到数据库中
        '''
        recs = CBrand.get_known_brands(start_idx, amount, 
                    sort_id, sort_type)
        brand_id = 1
        brands = []
        MBrand.clear_brands()
        for rec in recs:
            rec['brand_id'] = brand_id
            brand = {
                'brand_id': rec['brand_id'],
                'brand_name': rec['brand_name'],
                'brand_num': rec['brand_num']
            }
            brands.append(brand)
            brand_id += 1
            MBrand.insert(rec)
        data = {
            'total': len(brands),
            'brands': brands
        }
        return data
        
    @staticmethod
    def get_known_brands(start_idx=1, amount=-1, sort_id=1, 
                sort_type=1):
        print('获取已有品牌列表...')
        #
        base_dir = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw'
        bns = set()
        base_path = Path(base_dir)
        for brand_path in base_path.iterdir():
            brand_str = str(brand_path)
            arrs0 = brand_str.split('/')
            brand_name = arrs0[-1]
            bns.add(brand_name)
        brands = []
        for bn in bns:
            num = CBrand.get_files_num_in_folder('{0}/{1}'.format(base_dir, bn))
            brand = {'brand_name': bn, 'brand_num': num}
            brands.append(brand)
        return sorted(brands, key=CBrand.sort_by_num, reverse=False)












    @staticmethod
    def sort_by_num(item):
        return '{0:10d}_{1}'.format(item['brand_num'], item['brand_name'])

    @staticmethod
    def get_files_num_in_folder(folder_name):
        num = 0
        base_path = Path(folder_name)
        for model_path in base_path.iterdir():
            model_str = str(model_path)
            arrs0 = model_str.split('/')
            model_name = arrs0[-1]
            for year_path in model_path.iterdir():
                for file_obj in year_path.iterdir():
                    num += 1
        return num
    