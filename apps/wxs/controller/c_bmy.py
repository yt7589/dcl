#
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_bmy import MBmy
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


    