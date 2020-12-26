from flask import Flask

import sys
import os
sys.path.append(os.path.dirname(__file__))
import controller

app = Flask(__name__)


@app.route('/v1/api/rmbg')
def rmbg_api():
    return controller.request_img_from_api('/images/test.jpg')


@app.route('/v2/api/rmbg')
def rmbg_local():

    return controller.request_img_from_local('/images/test.jpg')


def run():
    app.run(debug=True)

