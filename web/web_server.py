# Web server for production mode and debug
import sys
import ctypes
import threading
import json
from flask import Flask
from flask import jsonify
from flask import request
import app_store

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.get_data(as_text=True))
    print('img_name: {0};'.format(data['img_name']))
    app_store.vars['lr'] -= 0.2
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})
    
class WebServer(threading.Thread):
    instance = None

    def __init__(self):
        threading.Thread.__init__(self)
        WebServer.instance = self
        self.name = 'web.WebServer'

    def run(self):
        app.run()

    def stop_thread(self):
        tid = ctypes.c_long(self.ident)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
        if 0 == res:
            raise ValueError("invalid thread id")
        else:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
