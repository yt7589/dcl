#

class BidVin(object):
    def __init__(self):
        self.name = 'apps.wxs.bid.BidVin'
        
    @staticmethod
    def get_vin_codes():
        '''
        获取Excel表格中所有车辆识别码列表
        '''
        vc_set = set()
        is_first_line = True
        with open('./logs/bid_20200708.csv', 'r', encoding='utf-8') as fd:
            for line in fd:
                line = line.strip()
                if not is_first_line:
                    arrs_a = line.split(',')
                    vin_code = arrs_a[-1]
                    vc_set.add(vin_code)
                else:
                    is_first_line = False
        return vc_set