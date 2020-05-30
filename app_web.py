import json
import flask
from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
#
from utils.ds_manager import DsManager

app = Flask(__name__)
CORS(app)
image_root = '/media/zjkj/35196947-b671-441e-9631-6245942d671b/yantao/web_root/images'

@app.route('/', methods=['GET'])
def ping_pong():
    return jsonify('Hello World!')     #（jsonify返回一个json格式的数据）

@app.route('/displayImage/<string:filename>', methods=['GET'])
def display_image(filename):
    global image_root
    if filename is None:
        return
    with open('{0}/{1}'.format(image_root,  filename), 'rb') as img_fd:
        image_data = img_fd.read()
    response = flask.make_response(image_data)
    response.headers['Content-Type'] = 'image/jpg'
    return response

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
    app.run(
        host = '0.0.0.0',
        port = 5000
    )

if __name__ == '__main__':
    main({})