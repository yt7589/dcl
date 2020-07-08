# 样本控制器类
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_sample import MSample

class CSample(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CSample'

    @staticmethod
    def add_sample(img_file, vin_id, bmy_id):
        MSample.is_sample_exists(img_file)
        iii = 1
        if 1 === iii:
            return 0
        if MSample.is_sample_exists(img_file):
            rec = MSample.get_sample_by_img_file(img_file)
            return rec['sample_id']
        sample_id = MPkGenerator.get_pk('sample_id')
        sample_vo = {
            'sample_id': sample_id,
            'img_file': img_file,
            'vin_id': vin_id,
            'bmy_id': bmy_id
        }
        rst = MSample.insert(sample_vo)
        return sample_id

    @staticmethod
    def get_vin_samples(vin_id):
        return MSample.get_vin_samples(vin_id)