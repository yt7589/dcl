import json
from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
#
from utils.ds_manager import DsManager

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def ping_pong():
    return jsonify('Hello World!')     #（jsonify返回一个json格式的数据）

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
            'total_num': 100
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