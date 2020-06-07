import json
import flask
from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
#
from utils.ds_manager import DsManager
from apps.vbg.vbg_util import VbgUtil
from apps.vbg.controller.c_vbg import CVbg
from apps.admin.controller.c_brand import CBrand
from apps.admin.controller.c_bmys import CBmys
from apps.admin.controller.c_ggh_bmy import CGghBmy

app = Flask(__name__)
CORS(app)
image_root = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/web_root/images'

# 导入公告号与品牌_车型_年款关系
@app.route('/addmin/gghToBmyDict', methods=['GET'])
def ggh_to_bmy_dict_api():
    return CGghBmy.ggh_to_bmy_dict_api()

# 获取品牌_车型_年款列表及每类中的图片数
@app.route('/admin/getBmys', methods=['GET'])
def get_bmys():
    return CBmys.get_bmys_api()

@app.route('/admin/getKnownBrands', methods=['GET'])
def get_known_brands():
    ''' 获取已知品牌列表 '''
    mode = request.args.get("mode")
    if '1' == mode:
        data = CBrand.get_known_brands_api()
    elif '2' == mode:
        data = CBrand.get_refresh_known_brands_api()
    resp = {
        'code': 0,
        'msg': 'Ok',
        'data': data
    }
    return json.dumps(resp, ensure_ascii=False)








@app.route('/', methods=['GET'])
def ping_pong():
    return jsonify('Hello World!')     #（jsonify返回一个json格式的数据）

@app.route('/getSurveyData', methods=['GET'])
def get_survey_data():
    survey_data = CVbg.get_survey_data(50)
    resp = {
        'code': 0,
        'msg': 'Ok',
        'data': {
            'survey_data': survey_data
        }
    }
    return json.dumps(resp, ensure_ascii=False)

def display_image_base(img_file):
    with open(img_file, 'rb') as img_fd:
        image_data = img_fd.read()
    response = flask.make_response(image_data)
    postfix = img_file[-4:]
    response.headers['Content-Type'] = 'image/{0}'.format(postfix)
    return response

@app.route('/displayVbicon/<string:filename>', methods=['GET'])
def display_vbicon(filename):
    global image_root
    if filename is None:
        return
    return display_image_base('{0}/vbg/{1}'.format(image_root,  filename))

@app.route('/displayImage/<string:filename>', methods=['GET'])
def display_image(filename):
    global image_root
    if filename is None:
        return
    return display_image_base('{0}/{1}'.format(image_root,  filename))

# /updateDict?folderName=/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao
@app.route('/updateDict', methods=['GET'])
def update_dict():
    folder_name = request.args.get("folderName")
    args = {'folder_name': folder_name}
    DsManager.run(DsManager.RUN_MODE_REFINE, args)
    resp = {
        'code': 0,
        'msg': 'Ok',
        'data': {
            'result': '更新字典成功！'
        }
    }
    return json.dumps(resp, ensure_ascii=False)

@app.route('/getBrands', methods=['GET'])
def get_brands():
    resp = {
        'code': 0,
        'msg': 'Ok',
        'data': {
            'users': [
                {'name': '闫涛', 'email': 'yt7589@qq.com'},
                {'name': 'Tom', 'email': 'tom@abc.com'}
            ],
            'total_num': 100
        }
    }
    return json.dumps(resp, ensure_ascii=False)

def main(args):
    i_debug = 10
    if 1 == i_debug:
        VbgUtil.get_data()
        return
    app.run(
        host = '0.0.0.0',
        port = 5000
    )

if __name__ == '__main__':
    main({})