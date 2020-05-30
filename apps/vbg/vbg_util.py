# 工具类：准备品牌及车标
from urllib.request import urlopen
from bs4 import BeautifulSoup

class VbgUtil(object):
    @staticmethod
    def get_data():
        html = urlopen(
            "http://www.chelogo.com/chebiao/list_1_1.html"
        ).read().decode('gb2312')
        bs = BeautifulSoup(html, "html.parser")
        items = bs.find_all("ul")
        for item in items:
            img_src = item.img['src']
            brand_name = item.img['alt']
            #img_src = img.attrs['src']
            #brand_name = img.attrs['alt'][:-2]
            print('{0}: {1};'.format(brand_name, img_src))