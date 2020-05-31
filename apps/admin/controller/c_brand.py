# 品牌控制器类
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
        with open('./work/bmy_to_fgvc_id_dict.txt', 'r', encoding='utf-8') as bfi_fd:
            raw_set = set()
            for line in bfi_fd:
                arrs0 = line.split(':')
                arrs1 = arrs0[0].split('_')
                brand_name = arrs1[0]
                raw_set.add(brand_name)
            brand_names = sorted(list(raw_set))
            #
            raw_set1 = set()
            base_path = Path('')
            for brand_path in base_path.iterdir():
                brand_str = str(brand_path)
                arrs0 = brand_str.split('/')
                brand_name = arrs0[-1]
                raw_set1.add(brand_name)
            print('directory num={0};'.format(len(raw_set1)))
            brands = brand_names
            rst = {
                'total': len(brands),
                'brands': brands
            }
        return rst