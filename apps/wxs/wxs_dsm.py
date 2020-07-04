# 数据集管理类，负责生成数据描述文件

class WxsDsm(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsDsm'

    @staticmethod
    def know_init_status():
        bid_brand_set, bid_bmy_set, bid_ggh_set = WxsDsm._get_bid_info()
        print('标书要求：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(bid_ggh_set), len(bid_brand_set), len(bid_bmy_set)
        ))
        our_brand_set, our_bmy_set, our_ggh_set = WxsDsm._get_our_info()
        print('自有情况：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(our_ggh_set), len(our_brand_set), len(our_bmy_set)
        ))
        # 所里有但是我们没有
        oh_brand_set = our_brand_set - bid_brand_set
        print('我们有所里没有品牌：{0}个；'.format(len(oh_brand_set)))
    @staticmethod
    def _get_our_info():
        print('掌握当前情况')
        brand_set = set()
        bmy_set = set()
        ggh_set = set()
        with open('./work/ggh_to_bmy_dict.txt', 'r', encoding='utf-8') as gfd:
            for line in gfd:
                row = line.strip()
                arrs0 = row.split(':')
                ggh_code = arrs0[0]
                ggh_set.add(ggh_code)
                bmy = arrs0[1]
                bmy_set.add(bmy)
                arrs1 = arrs0[1].split('_')
                brand_name = arrs1[0]
                brand_set.add(brand_name)
                model_name = arrs1[1]
                year_name = arrs1[2]
        return brand_set, bmy_set, ggh_set
    @staticmethod
    def _get_bid_info():
        brand_set = set()
        bmy_set = set()
        ggh_set = set()
        seq = 0
        with open('./logs/bid_20200708.csv', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                row = line.strip()
                print(row)
                arrs0 = row.split(',')
                if seq > 0:
                    brand_name = arrs0[2]
                    brand_set.add(brand_name)
                    model_name = arrs0[4]
                    year_name = arrs0[6]
                    bmy = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                    bmy_set.add(bmy)
                    ggh_code = arrs0[8]
                    ggh_set.add(ggh_code)
                seq += 1
        return brand_set, bmy_set, ggh_set

    