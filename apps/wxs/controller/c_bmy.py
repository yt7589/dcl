#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_bmy import MBmy
from apps.wxs.model.m_model import MModel
from apps.wxs.model.m_brand import MBrand
from apps.wxs.model.m_vin import MVin

class CBmy(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CBmy'

    @staticmethod
    def is_bmy_exists(bmy_name):
        return MBmy.is_bmy_exists(bmy_name)

    @staticmethod
    def add_bmy(bmy_name, bmy_code, brand_id, brand_code, model_id, model_code):
        if MBmy.is_bmy_exists(bmy_name):
            bmy_vo = MBmy.get_bmy_by_name(bmy_name)
            return int(bmy_vo['bmy_id'])
        bmy_id = MPkGenerator.get_pk('bmy_id')
        bmy_vo = {
            'bmy_id': bmy_id,
            'bmy_name': bmy_name,
            'bmy_code': bmy_code,
            'brand_id': brand_id,
            'brand_code': brand_code,
            'model_id': model_id,
            'model_code': model_code,
            'bmy_num': 1
        }
        rst = MBmy.insert(bmy_vo)
        return bmy_id

    @staticmethod
    def is_vin_exists(vin_code):
        return MVin.is_vin_exists(vin_code)

    @staticmethod
    def add_vin(vin_code, bmy_id, source_type):
        if MVin.is_vin_exists(vin_code):
            vin_vo = MVin.get_vin_by_code(vin_code)
            return int(vin_vo['vin_id'])
        vin_id = MPkGenerator.get_pk('vin_id')
        vin_vo = {
            'vin_id': vin_id,
            'vin_code': vin_code,
            'bmy_id': bmy_id,
            'source_type': source_type
        }
        rst = MVin.insert(vin_vo)
        return vin_id

    @staticmethod
    def get_bmy_id_by_vin_code(vin_code):
        rec = MVin.get_bmy_id_by_vin_code(vin_code)
        if rec:
            return int(rec['bmy_id']), int(rec['vin_id'])
        else: 
            return -1, -1

    @staticmethod
    def get_bmy_id_by_prefix_vin_code(vin_code):
        prefix_vin_code = vin_code[:8]
        recs = MVin.get_bmy_ids_by_vin_code(prefix_vin_code)
        if not recs:
            return -1, -1
        elif len(recs)>1:
            return -2, -2
        else:
            return int(recs[0]['bmy_id']), int(recs[0]['vin_id'])

    @staticmethod
    def get_vin_codes():
        return MVin.get_vin_codes()

    @staticmethod
    def get_vin_bmy_id_dict():
        recs = MVin.get_vin_bmy_id_dict()
        vin_bmy_id_dict = {}
        for rec in recs:
            vin_bmy_id_dict[rec['vin_code']] = int(rec['bmy_id'])
        return vin_bmy_id_dict

    @staticmethod
    def get_bmy_id_vin_dict():
        recs = MVin.get_vin_bmy_id_dict()
        bmy_id_vin_dict = {}
        for rec in recs:
            bmy_id_vin_dict[rec['bmy_id']] = rec['vin_code']
        return bmy_id_vin_dict
        
    @staticmethod
    def get_bmy_by_id(bmy_id):
        return MBmy.get_bmy_by_id(bmy_id)

    @staticmethod
    def get_bmy_id_bmy_name_dict():
        bmy_id_bmy_name_dict = {}
        recs = MBmy.get_bmy_id_bmy_names()
        for rec in recs:
            bmy_id_bmy_name_dict[int(rec['bmy_id'])] = rec['bmy_name']
        return bmy_id_bmy_name_dict
        
    @staticmethod
    def get_bmy_name_bmy_id_dict():
        bmy_name_bmy_id_dict = {}
        recs = MBmy.get_bmy_id_bmy_names()
        for rec in recs:
            bmy_name_bmy_id_dict[rec['bmy_name']] = int(rec['bmy_id'])
        return bmy_name_bmy_id_dict

    @staticmethod
    def get_bmy_id_bmy_vo_dict():
        '''
        求出t_bmy表的以bmy_id为键，以bmy_vo为值
        '''
        bmy_id_bmy_vo_dict = {}
        recs = MBmy.get_bmy_id_bmy_vos()
        for rec in recs:
            bmy_id_bmy_vo_dict[int(rec['bmy_id'])] = {
                'bmy_id': int(rec['bmy_id']),
                'bmy_name': rec['bmy_name'],
                'bmy_code': rec['bmy_code'],
                'brand_id': int(rec['brand_id']),
                'brand_code': rec['brand_code'],
                'model_id': int(rec['model_id']),
                'model_code': rec['model_code']
            }
        return bmy_id_bmy_vo_dict

    @staticmethod
    def get_vin_code_bmy_id_dict():
        recs = MVin.get_vin_code_bmy_id_dict()
        vin_code_bmy_id_dict = {}
        for rec in recs:
            vin_code_bmy_id_dict[rec['vin_code']] = int(rec['bmy_id'])
        return vin_code_bmy_id_dict

    @staticmethod
    def get_bmys():
        return MBmy.get_bmys()

    @staticmethod
    def get_vin_code_bmys():
        return MVin.get_vin_code_bmys()

    @staticmethod
    def get_bmy_id_brand_id_dict():
        recs = MBmy.get_bmy_id_brand_ids()
        bmy_id_brand_id_dict = {}
        for rec in recs:
            bmy_id_brand_id_dict[int(rec['bmy_id'])] = int(rec['brand_id'])
        return bmy_id_brand_id_dict

    @staticmethod
    def get_wxs_bmys():
        raw_bmys = MBmy.get_bmys()
        bmys = []
        for rb in raw_bmys:
            if not rb['bmy_code'].startswith('b'):
                bmy = {
                    'bmy_id': int(rb['bmy_id'])-1,
                    'bmy_name': rb['bmy_name'],
                    'bmy_code': rb['bmy_code']
                }
                bmys.append(bmy)
        return bmys

    @staticmethod
    def get_non_wxs_vins():
        return MVin.get_non_wxs_vins()

    @staticmethod
    def get_wxs_vins():
        return MVin.get_wxs_vins()

    @staticmethod
    def get_wxs_vin_code_bmy_id_dict():
        recs = MVin.get_wxs_vin_code_bmy_id_dict()
        wxs_vin_code_bmy_id_dict = {}
        for rec in recs:
            wxs_vin_code_bmy_id_dict[rec['vin_code']] = int(rec['bmy_id'])
        return wxs_vin_code_bmy_id_dict

    @staticmethod
    def get_bmy_id_bm_vo_dict():
        '''
        获取bmy_id（年款头输出）与车型值对象的字典
        '''
        bmy_id_bm_vo_dict = {}
        bmy_id_model_ids = MBmy.get_bmy_id_model_ids()
        for bimi in bmy_id_model_ids:
            bmy_id = int(bimi['bmy_id'])
            model_id = int(bimi['model_id'])
            model_vo = MModel.get_model_vo_by_id(model_id)
            bm_vo = {
                'model_id': model_id,
                'model_name': model_vo['model_name'],
                'model_code': model_vo['model_code'],
                'source_type': model_vo['source_type']
            }
            bmy_id_bm_vo_dict[bmy_id] = bm_vo
        return bmy_id_bm_vo_dict

    @staticmethod
    def get_bmy_code_to_bmy_id_dict():
        bmy_code_to_bmy_id_dict = {}
        recs = MBmy.get_bmy_code_to_bmy_id_dict()
        for rec in recs:
            bmy_code_to_bmy_id_dict[rec['bmy_code']] = int(rec['bmy_id'])
        return bmy_code_to_bmy_id_dict

    @staticmethod
    def update_bmy_codes(bmy_id, bmy_code, bm_code, brand_code):
        MBmy.update_bmy_codes(bmy_id, bmy_code, bm_code, brand_code)

    @staticmethod
    def get_vin_id_codes():
        return MVin.get_vin_id_codes()

    @staticmethod
    def update_vin_code_by_vin_id(vin_id, vin_code):
        MVin.update_vin_code_by_vin_id(vin_id, vin_code)

    @staticmethod
    def get_bmy_id_2_bmy_vo():
        raw_bmys = MBmy.get_bmys()
        bmy_id_2_bmy_vo = {}
        for rb in raw_bmys:
            if not rb['bmy_code'].startswith('b'):
                bmy_vo = {
                    'bmy_id': int(rb['bmy_id']),
                    'bmy_name': rb['bmy_name'],
                    'bmy_code': rb['bmy_code'],
                    'brand_id': int(rb['brand_id']),
                    'model_id': int(rb['model_id'])
                }
                bmy_id_2_bmy_vo[int(rb['bmy_id'])] = bmy_vo
        return bmy_id_2_bmy_vo
        
    @staticmethod
    def get_zjkj_bmys():
        items = []
        bmys = CBmy.get_bmys()
        for bmy in bmys:
            brand_id = int(bmy['brand_id'])
            brand_vo = MBrand.get_brand_vo_by_id(brand_id)
            model_id = int(bmy['model_id'])
            model_vo = MModel.get_model_vo_by_id(model_id)
            item = {
                'bmy_id': int(bmy['bmy_id']),
                'bmy_code': bmy['bmy_code'],
                'bmy_name': bmy['bmy_name'],
                'model_id': int(bmy['model_id']),
                'model_code': bmy['model_code'],
                'model_name': model_vo['model_name'],
                'brand_id': int(bmy['brand_id']),
                'brand_code': bmy['brand_code'],
                'brand_name': brand_vo['brand_name']
            }
            items.append(item)
        return items
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            