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
            brand = {'name': bn, 'num': num}
            brands.append(brand)
        return sorted(brands, key=CBrand.sort_by_num, reverse=False)










    @staticmethod
    def sort_by_num(item):
        return item['num']

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
    