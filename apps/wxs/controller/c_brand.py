#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_brand import MBrand

class CBrand(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CBrand'

    def add_brand(brand_name, brand_code):
        brand_id = MPkGenerator.get_pk('brand_id')
        brand_vo = {
            'brand_id': brand_id,
            'brand_name': brand_name,
            'brand_code': brand_code,
            'brand_num': 1
        }
        rst = MBrand.insert(brand_vo)
        print('insert rst={0};'.format(rst))