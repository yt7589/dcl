# 数据源控制器类
from pathlib import Path
from apps.admin.controller.c_bmy import CBmy
from apps.admin.controller.c_brand import CBrand
from apps.admin.controller.c_model import CModel
from apps.admin.controller.c_ggh_bmy import CGghBmy
from apps.admin.controller.c_vehicle_image import CVehicleImage
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_data_source import MDataSource

class CDataSource(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CDataSource'

    @staticmethod
    def import_data():
        # 处理单个目录
        base_path = Path('/media/zjkj/35196947-b671-441e-9631-6245942d671b/fgvc_dataset/raw')
        CDataSource.import_folder_images(base_path)

    num = 0
    @staticmethod
    def import_folder_images(base_path):
        unknown_ggh = set()
        for item_obj in base_path.iterdir():
            item_str = str(item_obj)
            if item_obj.is_dir():
                CDataSource.import_folder_images(item_obj)
            elif item_str.endswith(('jpg','png','jpeg','bmp')):
                arrs0 = item_str.split('/')
                img_file = arrs0[-1]
                if img_file.startswith('白') or img_file.startswith('夜'):
                    print('处理测试数据集文件：{0};'.format(img_file))
                    # 获取品牌车型年款
                    arrs1 = img_file.split('_')
                    brand_name = arrs1[3]
                    model_name = arrs1[4]
                    bmy_name = arrs1[5]
                    bmy_name = '{0}_{1}_{2}'.format(brand_name, model_name, bmy_name)
                    print('       品牌车型年款: {0};'.format(bmy_name))
                    # 如果没有品牌车型年款，则向mongodb添加品牌车型年款
                    bmy_id = CBmy.find_bmy_id_by_name(bmy_name)
                    # 添加到t_vehicle_image表
                    vehicle_image_id = CVehicleImage.add_vehicle_image(item_str)
                    # 添加到t_data_source表，其类型为测试数据集
                else:
                    print('处理训练数据集文件：{0};'.format(img_file))
                    # 获取公告号
                    arrs2 = img_file.split('_')
                    arrs3 = arrs2[0].split('#')
                    ggh_code = arrs3[0]
                    print('       公告号：{0};'.format(ggh_code))
                    # 获取品牌车型年款bmy_id，如果没有则记录下该文件并continue
                    bmy_id = CGghBmy.get_bmy_id_by_ggh_code(ggh_code)
                    if bmy_id < 0:
                        print('未知公告号：{0};'.format(ggh_code))
                        unknown_ggh.add('{0}:{1}'.format(ggh_code, item_str))
                        continue
                    # 添加到t_vehicle_image表
                    vehicle_image_id = CVehicleImage.add_vehicle_image(item_str)
                    # 添加到t_data_source表
                if CDataSource.num > 30:
                    break
            CDataSource.num += 1
            with open('./logs/unknown_ggh_2.txt', 'a+', encoding='utf-8') as ug_fd:
                for g in unknown_ggh:
                    print('### {0};'.format(g))
                    ug_fd.write('{0}\n'.format(g))

    @staticmethod
    def add_data_source_sample(vehicle_image_id, bmy_id, type):
        rec = MDataSource.get_data_source_by_vid(vehicle_image_id, bmy_id)
        if rec is None:
            data_source_id = MPkGenerator.get_pk('data_source')
            data_source_vo = {
                'data_source_id': data_source_id,
                'vehicle_image_id': vehicle_image_id,
                'bmy_id': bmy_id,
                'state': 0,
                'type': type
            }
        else:
            data_source_id = rec['data_source_id']
        return data_source_id