from flask import Flask, request
import json
import numpy
import sys
import os
import cv2
import base64

sys.path.append(os.path.dirname(__file__))
import controller

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


@app.route('/v1/api/rmbg')
def rmbg_api():
    return controller.request_img_from_api('/images/test.jpg')


@app.route('/v2/api/rmbg', methods=['POST'])
def rmbg_local():
    binary = request.json['buf']
    img_url = ''

    if binary == '' and img_url == '':
        return {'status': 'needs binary or img_url request param '}
    return controller.request_img_from_local(binary, img_url)


@app.route('/v2/api/rmbg_m', methods=['POST'])
def rmbg_local_mchine():
    binary = request.json['buf']
    img_url = ''

    if binary == '' and img_url == '':
        return {'status': 'needs binary or img_url request param '}
    return controller.request_img_from_local(binary, img_url)


@app.route('/v2/api/test', methods=['GET'])
def test_local():
    img = cv2.imread('./images/test4.jpg')
    result, dst_data = cv2.imencode('.png', img)
    binary = base64.b64encode(dst_data)

    img_url = ''

    if binary == '' and img_url == '':
        return {'status': 'needs binary or img_url request param '}
    return controller.request_img_from_local_machine(binary, img_url)

@app.route('/')
def hello():
    return 'hello'


def run():
    app.run(debug=True)
