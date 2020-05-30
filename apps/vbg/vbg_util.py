# 工具类：准备品牌及车标
from urllib.request import urlopen

class VbgUtil(object):
    @staticmethod
    def get_data():
        html = urlopen(
            "http://www.chelogo.com/chebiao/list_1_1.html"
        ).read().decode('utf-8')
        print(html)