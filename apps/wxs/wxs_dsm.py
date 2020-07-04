# 数据集管理类，负责生成数据描述文件
from apps.wxs.model.m_pk_generator import MPkGenerator

class WxsDsm(object):
    def __init__(self):
        self.name = 'apps.wxs.WxsDsm'

    @staticmethod
    def test_get_pk():
        brand_id = MPkGenerator.get_pk('brand_id')
        print('brand_id={0};'.format(brand_id))

    @staticmethod
    def know_init_status():
        i_debug = 1
        WxsDsm.test_get_pk()
        if 1 == i_debug:
            return

        bid_brand_set, bid_bmy_set, bid_vin_set = WxsDsm._get_bid_info()
        print('标书要求：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(bid_vin_set), len(bid_brand_set), len(bid_bmy_set)
        ))
        our_brand_set, our_bmy_set, our_vin_set = WxsDsm._get_our_info()
        print('自有情况：车辆识别码：{0}个；品牌：{1}个；年款：{2}个；'.format(
            len(our_vin_set), len(our_brand_set), len(our_bmy_set)
        ))
        print('******************************************************')
        # 统计品牌情况
        common_brand_set = our_brand_set & bid_brand_set
        print('我们和标书公共品牌：{0}个'.format(len(common_brand_set)))
        oh_brand_set = our_brand_set - bid_brand_set
        print('我们有标书没有品牌：{0}个；'.format(len(oh_brand_set)))
        WxsDsm.write_set_to_file(oh_brand_set, './logs/we_had_brand.txt')
        bh_brand_set = bid_brand_set - our_brand_set
        print('标书有我们没有的品牌：{0}个'.format(len(bh_brand_set)))
        WxsDsm.write_set_to_file(bh_brand_set, './logs/bid_had_brand.txt')
        all_brand_set = our_brand_set | bid_brand_set
        print('共有品牌：{0}个'.format(len(all_brand_set)))
        print('******************************************************')
        # 统计年款情况
        common_bmy_set = our_bmy_set & bid_bmy_set
        print('我们和标书公共年款：{0}个'.format(len(common_bmy_set)))
        oh_bmy_set = our_bmy_set - bid_bmy_set
        print('我们有标书没有年款：{0}个'.format(len(oh_bmy_set)))
        WxsDsm.write_set_to_file(oh_bmy_set, './logs/we_had_bmy.txt')
        bh_bmy_set = bid_bmy_set - our_bmy_set
        print('标书有我们没有年款：{0}个'.format(len(bh_bmy_set)))
        WxsDsm.write_set_to_file(bh_bmy_set, './logs/bid_had_bmy.txt')
        all_bmy_set = our_bmy_set | bid_bmy_set
        print('共有年款：{0}个'.format(len(all_bmy_set)))
        print('******************************************************')
        # 统计车辆识别码情况
        common_vin_set = our_vin_set & bid_vin_set
        print('我们和标书共有车辆识别码：{0}个'.format(len(common_vin_set)))
        oh_vin_set = our_vin_set - bid_vin_set
        print('我们有标书没有车辆识别码：{0}个'.format(len(oh_vin_set)))
        WxsDsm.write_set_to_file(oh_vin_set, './logs/we_had_vin.txt')
        bh_vin_set = bid_vin_set - our_vin_set
        print('标书有我们没有车辆识别码：{0}个'.format(len(bh_vin_set)))
        WxsDsm.write_set_to_file(bh_vin_set, './logs/bid_had_vin.txt')
        all_vin_set = our_vin_set | bid_vin_set
        print('共有车辆识别码：{0}个'.format(len(all_vin_set)))
    @staticmethod
    def _get_our_info():
        print('掌握当前情况')
        brand_set = set()
        bmy_set = set()
        vin_set = set()
        with open('./work/ggh_to_bmy_dict.txt', 'r', encoding='utf-8') as gfd:
            for line in gfd:
                row = line.strip()
                arrs0 = row.split(':')
                vin_code = arrs0[0]
                vin_set.add(vin_code)
                raw_bmy = arrs0[1]
                arrs1 = raw_bmy.split('_')
                bmy = '{0}牌_{1}_{2}'.format(arrs1[0], arrs1[1], arrs1[2])
                bmy_set.add(bmy)
                arrs1 = arrs0[1].split('_')
                brand_name = arrs1[0]
                brand_set.add('{0}牌'.format(brand_name))
                model_name = arrs1[1]
                year_name = arrs1[2]
        return brand_set, bmy_set, vin_set
    @staticmethod
    def _get_bid_info():
        brand_set = set()
        bmy_set = set()
        vin_set = set()
        seq = 0
        with open('./logs/bid_20200708.csv', 'r', encoding='utf-8') as cfd:
            for line in cfd:
                row = line.strip()
                arrs0 = row.split(',')
                if seq > 0:
                    brand_name = arrs0[2]
                    brand_set.add(brand_name)
                    model_name = arrs0[4]
                    year_name = arrs0[6]
                    bmy = '{0}_{1}_{2}'.format(brand_name, model_name, year_name)
                    bmy_set.add(bmy)
                    vin_code = arrs0[8]
                    vin_set.add(vin_code)
                seq += 1
        return brand_set, bmy_set, vin_set

    @staticmethod
    def write_set_to_file(set_obj, filename):
        ''' 将集合内容写到文件中，写入前先对集合内容进行排序 '''
        lst = list(set_obj)
        lst.sort()
        with open(filename, 'w+', encoding='utf-8') as wfd:
            for item in lst:
                wfd.write('{0}\n'.format(item))

    