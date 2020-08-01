#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_brand import MBrand

class CBrand(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CBrand'

    @staticmethod
    def add_brand(brand_name, brand_code, source_type):
        if MBrand.is_brand_exists(brand_name):
            return
        brand_id = MPkGenerator.get_pk('brand_id')
        brand_vo = {
            'brand_id': brand_id,
            'brand_name': brand_name,
            'brand_code': brand_code,
            'source_type': source_type,
            'brand_num': 1
        }
        rst = MBrand.insert(brand_vo)
        return True

    @staticmethod
    def get_brand_by_name(brand_name):
        '''
        根据品牌名称求出品牌详细信息
        '''
        return MBrand.get_brand_by_name(brand_name)

    @staticmethod
    def get_wxs_brands():
        return MBrand.get_wxs_brands()