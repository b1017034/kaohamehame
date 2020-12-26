import cv2
import os
import base64
import numpy as np
import requests
import matplotlib.pyplot as plt
import tempfile


def is_nan_or_inf(num: float) -> bool:
    return np.isnan(num) \
           and np.isnan(num) \
           and np.isinf(num) \
           and np.isinf(num)


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
        img_binary = base64.b64decode(binary)
        nparr = np.frombuffer(img_binary, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    print(os.path.dirname(__file__))
    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2値化
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # マスク生成
    mask = np.zeros(edges.shape)

    # 特徴検出
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ellipses = []
    for i, cnt in enumerate(contours):
        # 楕円フィッティング
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)

            # TODO: 良しなに
            if 0.9 <= ellipse[1][0] / ellipse[1][1] <= 1.2 and 40 <= ellipse[1][1] <= 400:
                print(ellipse)
                # resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                if not is_nan_or_inf(ellipse[0][0]) and not is_nan_or_inf(ellipse[0][1]):
                    # 楕円描画
                    ellipses.append(
                        {
                            'points': {'x': ellipse[0][0], 'y': ellipse[0][1]},
                            'axis': {'long': ellipse[1][0], 'short': ellipse[1][1]},
                            'radian': ellipse[2]
                        }
                    )
                    mask = cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
    print(ellipses)
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

    return {'img': image_to_base64('tmp.png'), 'ellipses': ellipses}


if __name__ == '__main__':
    img = remove_bg('../images/panel1.jpg')

    plt.imshow(img['img'])
    plt.show()
