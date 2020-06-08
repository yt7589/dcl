# 数据源控制器类
from pathlib import Path

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
        for item_obj in base_path.iterdir():
            item_str = str(item_obj)
            if item_obj.is_dir():
                CDataSource.import_folder_images(item_obj)
            elif item_str.endswith(('jpg','png','jpeg','bmp')):
                print('导入图片：{0};'.format(item_str))
                if CDataSource.num > 30:
                    break
            CDataSource.num += 1