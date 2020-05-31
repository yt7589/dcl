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
            base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
            for brand_path in base_path.iterdir():
                brand_str = str(brand_path)
                arrs0 = brand_str.split('/')
                brand_name = arrs0[-1]
                raw_set1.add(brand_name)
            # 
            raw_set2 = set()
            with open('./work/ggh_to_bmy_dict.txt', 'r', encoding='utf-8') as gb_fd:
                for line in gb_fd:
                    arrs0 = line.split(':')
                    arrs1 = arrs0[1].split('_')
                    brand_name = arrs1[0]
                    raw_set2.add(brand_name)
            print('raw_set2={0};'.format(len(raw_set2)))
            #
            raw_set3 = set()
            base_path1 = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/t1')
            for brand_path in base_path1.iterdir():
                brand_str = str(brand_path)
                arrs0 = brand_str.split('/')
                brand_name = arrs0[-1]
                raw_set3.add(brand_name)
            print('raw_set3={0};'.format(len(raw_set3)))
            #
            print('directory num={0};'.format(len(raw_set1)))
            brands = brand_names
            rst = {
                'total': len(brands),
                'brands': brands
            }
        return rst