# Web server for production mode and debug
import sys
import ctypes
import threading
import json
from flask import Flask
from flask import jsonify
from flask import request
import app_store
from web.dcl_classifier import DclClassifier

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.get_data(as_text=True))
    class_id, class_name = WebServer.classifier.predict(data['img_name'])
    print('img_name: {0} => {1};'.format(data['img_name'], class_id))
    return jsonify({'class_id': class_id, 'class_name': class_name})
    
class WebServer(threading.Thread):
    classifier = None

    def __init__(self):
        threading.Thread.__init__(self)
        WebServer.instance = self
        self.name = 'web.WebServer'
        WebServer.classifier = DclClassifier()

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
