# 无锡所测试工具：
# 指定wxs_ds_rst.txt文件位置
# 将整理过(./support/wxs_ds)的无锡所测试数据集图片放到指定目录下
# 通过client1.8调用车辆检测接口，将结果保存在指定目录下
# 运行本程序计算精度
import json
from pathlib import Path

class WxsBidTest(object):
    def __init__(self):
        self.refl = 'utils.WxsBidTest'
        
    @staticmethod
    def get_wxs_bid_ds_scores():
        print('计算在无锡所数据集上的品牌精度和车型精度工具类 v0.0.1')
        # 指定wxs_ds_rst.txt文件位置，该文件格式为：图片文件名*品牌代码*车型代码
        wxs_ds_rst_txt = './support/wxs_ds_rst.txt'
        vd_json_folder = './support/wxs_ds_rst'
        brand_acc_dict = {}
        bm_acc_dict = {}
        with open(wxs_ds_rst_txt, 'r', encoding='utf-8') as rfd:
            for line in rfd:
                line = line.strip()
                arrs_a = line.split('*')
                img_file = arrs_a[0]
                brand_code = arrs_a[1]
                bm_code = arrs_a[2]
                brand_acc_dict[img_file] = brand_code
                bm_acc_dict[img_file] = bm_code
        base_path = Path(vd_json_folder)
        brand_num = 0
        bm_num = 0
        total_num = 0
        for fo in base_path.iterdir():
            full_fn = str(fo)
            arrs_a = full_fn.split('/')
            json_file = arrs_a[-1]
            arrs_b = json_file.split('_')
            img_file = '{0}_{1}_{2}_{3}_{4}'.format(
                arrs_b[0], arrs_b[1], arrs_b[2], arrs_b[3], arrs_b[4]
            )
            print('full_fn: {0}'.format(full_fn))
            with open(full_fn, 'r', encoding='utf-8') as jfd:
                vd_json = jfd.read()
            data = json.loads(vd_json) # VdJsonManager.get_img_reid_feature_vector(full_fn)
            clpp, ppcx, ppxhms = WxsDsm.parse_vd_json_for_ppcx(data)
            brand_code = brand_acc_dict[img_file]
            bm_code = bm_acc_dict[img_file]
            total_num += 1
            if brand_code == clpp:
                brand_num += 1
            if bm_code == ppcx:
                bm_num += 1
        print('brand_acc: {0};'.format(brand_num / total_num))
        print('bm_acc: {0};'.format(bm_num / total_num))
            
    @staticmethod
    def parse_vd_json_for_ppcx(data):
        cllxfls = ['11', '12', '13', '14', '21', '22']
        if ('VEH' not in data) or len(data['VEH']) < 1:
            return None, None, None
        else:
            # 找到面积最大的检测框作为最终检测结果
            max_idx = -1
            max_area = 0
            for idx, veh in enumerate(data['VEH']):
                cllxfl = veh['CXTZ']['CLLXFL'][:2]
                if cllxfl in cllxfls:
                    box_str = veh['WZTZ']['CLWZ']
                    arrs_a = box_str.split(',')
                    x1, y1, w, h = int(arrs_a[0]), int(arrs_a[1]), int(arrs_a[2]), int(arrs_a[3])
                    area = w * h
                    if area > max_area:
                        max_area = area
                        max_idx = idx
            if max_idx < 0:
                return None, None, None
            else:
                return data['VEH'][max_idx]['CXTZ']['CLPP'], data['VEH'][max_idx]['CXTZ']['PPCX'], \
                        data['VEH'][max_idx]['CXTZ']['PPXHMS']
                        
def main(args={}):
    WxsBidTest.get_wxs_bid_ds_scores()
    
if '__main__' == __name__:
    main()