# 工具类：准备品牌及车标
from urllib.request import urlopen

class VbgUtil(object):
    @staticmethod
    def get_data():
        html = urlopen(
            "https://mp.csdn.net/postedit"
        ).read().decode('utf-8')
        print(html)