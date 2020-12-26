import requests
import os
import ast
import base64
import sys

import numpy
import json


sys.path.append(os.path.dirname(__file__))

from trimap import main


def request_img_from_api(path: str) -> dict:
    print(os.path)
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(os.getcwd() + path, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': 'INSERT_YOUR_API_KEY_HERE'},
    )
    if response.status_code == requests.codes.ok:
        # success
        return {
            'status': response.status_code,
            'img': response.content,
        }
        # with open('no-bg.png', 'wb') as out:
        #     out.write(response.content)
    else:
        # fail
        print("Error:", response.status_code, response.text)

        return ast.literal_eval(response.text)


def request_img_from_local(path: str) -> dict:
    img_dic = main.remove_bg(path)
    # img_bin =

    # img_base64 = base64.b64encode(f.read()).decode('utf-8')
    return img_dic


