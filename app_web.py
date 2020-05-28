import json
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def ping_pong():
    return jsonify('Hello World!')     #（jsonify返回一个json格式的数据）

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
    app.run()

if __name__ == '__main__':
    main({})