# 数据集样本控制器类
from apps.wxs.model.m_pk_generator import MPkGenerator
from apps.wxs.model.m_dataset import MDataset
from apps.wxs.model.m_dataset_sample import MDatasetSample

class CDataset(object):
    def __init__(self):
        self.name = 'apps.wxs.controller.CDataset'

    @staticmethod
    def add_dataset_sample(dataset_id, sample_id, sample_type):
        if MDatasetSample.is_dataset_sample_exists(dataset_id, sample_id, sample_type):
            rec = MDatasetSample.get_dataset_sample_by_infos(dataset_id, sample_id, sample_type)
            return rec['dataset_sample_id']
        dataset_sample_id = MPkGenerator.get_pk('dataset_sample_id')
        dataset_sample_vo = {
            'dataset_sample_id': dataset_sample_id,
            'dataset_id': dataset_id,
            'sample_id': sample_id,
            'sample_type': sample_type
        }
        rst = MDatasetSample.insert(dataset_sample_vo)
        return dataset_sample_id