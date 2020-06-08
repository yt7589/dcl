# 车辆图片控制器
from apps.admin.model.m_pk_generator import MPkGenerator
from apps.admin.model.m_vehicle_image import MVehicleImage

class CVehicleImage(object):
    def __init__(self):
        self.name = 'apps.admin.controller.CVehicleImage'

    @staticmethod
    def get_vehicle_image_vo(filename):
        '''
        根据文件名求出文件车辆图片编号和全路径文件名
        '''
        return MVehicleImage.get_vehicle_image_vo(filename)

    @staticmethod
    def add_vehicle_image(image_file):
        '''
        将图片文件保存到t_vehicle_image表中
        参数：image_file 全路径图片文件名
        返回值：vehicle_image_id
        '''
        arrs0 = image_file.split('/')
        filename = arrs0[-1]
        rec = CVehicleImage.get_vehicle_image_vo(filename)
        if rec is None:
            vehicle_image_id = MPkGenerator.get_pk('vehicle_image')
            vehicle_image_vo = {
                'vehicle_image_id': vehicle_image_id,
                'filename': filename,
                'full_path': image_file
            }
            MVehicleImage.insert(vehicle_image_vo)
        else:
            vehicle_image_id = rec['vehicle_image_id']
        return vehicle_image_id
