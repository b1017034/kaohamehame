from flask import Flask
import json
import numpy
import sys
import os
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


@app.route('/v2/api/rmbg')
def rmbg_local():

    return controller.request_img_from_local('./images/panel1.jpg')


def run():
    app.run(debug=True)

