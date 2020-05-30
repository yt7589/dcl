# 工具类：准备品牌及车标
import time
import json
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
#
from apps.vbg.model.m_vehicle_brand import MVehicleBrand

class VbgUtil(object):
    @staticmethod
    def get_data():
        '''
        html = urlopen(
            "http://www.chelogo.com/chebiao/list_1_3.html"
        ).read().decode('gb2312')
        bs = BeautifulSoup(html, "html.parser")
        '''
        with open('/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/web_root/b3.html') as fd:
            bs = BeautifulSoup(fd, "html.parser")
        items = bs.find_all("ul")
        model = MVehicleBrand()
        vehicle_brand_id = 101
        for idx in range(1, len(items)):
            vehicle_brand_vo = {'vehicle_brand_id': vehicle_brand_id}
            img = items[idx].contents[1].contents[0]
            vehicle_brand_vo['vehicle_brand_name'] = img['alt'][:-2]
            vehicle_brand_vo['vbicon'] = img['src']
            notes = str(items[idx].contents[5])
            arrs0 = notes.split('　')
            vehicle_brand_vo['vehicle_brand_alias'] = arrs0[0][6:]
            vehicle_brand_vo['place_of_origin'] = arrs0[1][3:]
            print('     {0}; {1:03d}'.format(json.dumps(vehicle_brand_vo, ensure_ascii=False), vehicle_brand_id))
            model.insert_vehicle_brand(vehicle_brand_vo)
            VbgUtil.download_image(vehicle_brand_vo['vbicon'], 
                        '/media/zjkj/35196947-b671-441e-9631-6245942d671b/'
                        'yantao/web_root/images/vbg/vbicon_{0:03d}.jpg'.format(vehicle_brand_id))
            vehicle_brand_id += 1
            time.sleep(1.0)

    @staticmethod
    def download_image(img_url, img_file):
        urllib.request.urlretrieve(img_url, img_file)