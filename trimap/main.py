import cv2
import os
import base64
import numpy as np
import requests
import matplotlib.pyplot as plt
import tempfile
import csv
import pickle
import pandas as pd
import sklearn
from datetime import datetime as dt


def is_nan_or_inf(num: dict) -> bool:
    if is_nan_or_inf_float(num[0][0]):
        return True
    if is_nan_or_inf_float(num[1][0]):
        return True
    if is_nan_or_inf_float(num[0][1]):
        return True
    if is_nan_or_inf_float(num[1][1]):
        return True
    return False


def is_nan_or_inf_float(num: float) -> bool:
    return np.isnan(num) or np.isinf(num)

def ndarray_to_base64(dst):
    result, dst_data = cv2.imencode('.png', dst)
    dst_base64 = base64.b64encode(dst_data).decode('utf-8')

    return dst_base64


def image_to_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def base64_to_image(binary):
    imgdata = base64.b64decode(binary)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)

def url_to_numpy(url):
    res = requests.get(url)
    img = None

    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img


def readb64(binary):
    img_binary = base64.b64decode(binary)
    nparr = np.fromstring(img_binary, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)


def check_kaopane(ellipse, w, h):
    check = True
    # 円に近くない
    if 0.9 >= ellipse[1][0] / ellipse[1][1] >= 1.2:
        check = False

    # 円の大きさ
    if 40 >= ellipse[1][1] >= 400:
        check = False

    # 回転（0 or 90 に近くない）
    if 30 <= ellipse[3] <= 330 :
        check = False


def remove_bg(binary: str = None, img_url='') -> dict:
    img: None = None
    print('read')
    print(type(binary))

    # 画像読み込み
    if binary == '':
        img = url_to_numpy(img_url)
    elif img_url == '':
        base64_to_image(binary)
        img = cv2.imread('some_image.jpg')
        plt.imshow(img)
        plt.show()
    else:
        base64_to_image(binary)
        img = cv2.imread('some_image.jpg')
        plt.imshow(img)
        plt.show()

    print(os.path.dirname(__file__))
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2値化
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # 結果画像の黒の部分を灰色にする。
    bimg = edges // 4 + 255 * 3 // 4
    resimg = cv2.merge((bimg, bimg, bimg))

    # マスク生成
    mask = np.zeros(edges.shape)

    # 特徴検出
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ellipses = []

    tdatetime = dt.now()
    with open('../csv/svm.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for i, cnt in enumerate(contours):
            # 楕円フィッティング
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                # print(i + 1, ellipse)

                if not is_nan_or_inf(ellipse):
                    print(i + 1, ellipse)
                    with open('model.pickle.god', mode='rb') as fp:
                        clf = pickle.load(fp)
                    pf = np.array([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]]).reshape(1, -1)
                    test = clf.predict(pf)
                    if test >= 0.5:
                        print(test)

                    if 0.65 <= ellipse[1][0] / ellipse[1][1] <= 1.7 and 40 <= ellipse[1][1] <= 400:
                        # writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.0])
                        cx = int(ellipse[0][0])
                        cy = int(ellipse[0][1])
                        resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                        cv2.drawMarker(resimg, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                                       thickness=1)
                        cv2.putText(resimg, str(i + 1), (cx + 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255),
                                    1,
                                    cv2.LINE_AA)

                        # 楕円描画
                        ellipses.append(
                            {
                                'points': {'x': ellipse[0][0], 'y': ellipse[0][1]},
                                'axis': {'long': ellipse[1][0], 'short': ellipse[1][1]},
                                'radian': ellipse[2]
                            }
                        )
                        mask = cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
                        #writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.5])
                    #else:
                        #writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.0])

    # maskの反転
    mask = 255 - mask

    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    # masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    # masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    c_blue, c_green, c_red = cv2.split(img)

    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))


    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_a)
    fig.savefig('tmp.png')

    plt.imshow(resimg)
    plt.show()

    return {'img': image_to_base64('tmp.png'), 'ellipses': ellipses}


def remove_bg_machine(binary: str = None, img_url='') -> dict:
    img: None = None
    print('read')
    print(type(binary))

    # 画像読み込み
    if binary == '':
        img = url_to_numpy(img_url)
    elif img_url == '':
        base64_to_image(binary)
        img = cv2.imread('some_image.jpg')
        plt.imshow(img)
        plt.show()
    else:
        base64_to_image(binary)
        img = cv2.imread('some_image.jpg')
        plt.imshow(img)
        plt.show()

    print(os.path.dirname(__file__))
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2値化
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # 結果画像の黒の部分を灰色にする。
    bimg = edges // 4 + 255 * 3 // 4
    resimg = cv2.merge((bimg, bimg, bimg))

    # マスク生成
    mask = np.zeros(edges.shape)

    # 特徴検出
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ellipses = []

    tdatetime = dt.now()
    with open('../csv/svm.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for i, cnt in enumerate(contours):
            # 楕円フィッティング
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                # print(i + 1, ellipse)

                if not is_nan_or_inf(ellipse):
                    print(i + 1, ellipse)
                    with open('model.pickle.god', mode='rb') as fp:
                        clf = pickle.load(fp)
                    pf = np.array([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]]).reshape(1, -1)
                    test = clf.predict(pf)
                    if test >= 0.5:
                        print(test)

                    #if 0.65 <= ellipse[1][0] / ellipse[1][1] <= 1.7 and 40 <= ellipse[1][1] <= 400:
                        # writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.0])
                        cx = int(ellipse[0][0])
                        cy = int(ellipse[0][1])
                        resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                        cv2.drawMarker(resimg, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10,
                                       thickness=1)
                        cv2.putText(resimg, str(i + 1), (cx + 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255),
                                    1,
                                    cv2.LINE_AA)

                        # 楕円描画
                        ellipses.append(
                            {
                                'points': {'x': ellipse[0][0], 'y': ellipse[0][1]},
                                'axis': {'long': ellipse[1][0], 'short': ellipse[1][1]},
                                'radian': ellipse[2]
                            }
                        )
                        mask = cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
                        #writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.5])
                    #else:
                        #writer.writerow([ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2], 0.0])

    # maskの反転
    mask = 255 - mask

    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    # masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    # masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    c_blue, c_green, c_red = cv2.split(img)

    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))


    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_a)
    fig.savefig('tmp.png')

    plt.imshow(resimg)
    plt.show()

    return {'img': image_to_base64('tmp.png'), 'ellipses': ellipses}


if __name__ == '__main__':
    img = cv2.imread('../images/slack-imgs.jpg')
    result, dst_data = cv2.imencode('.png', img)
    binary = base64.b64encode(dst_data)
    remove_bg(binary, '')
