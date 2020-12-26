import requests
import os
import ast

import sys

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


def request_img_from_local(binary='', img_url='') -> dict:
    img_dic = main.remove_bg(binary, img_url)
    print(img_dic['img'])
    return img_dic


