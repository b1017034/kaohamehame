import cv2
import os
import base64
import numpy as np
import matplotlib.pyplot as plt

def isNanorInf(num: float) -> bool:
    return np.isnan(num) \
           and np.isnan(num) \
           and np.isinf(num) \
           and np.isinf(num)


def NdarrayToBase64(dst):
    result, dst_data = cv2.imencode('.jpg', dst)
    dst_base64 = base64.b64encode(dst_data).decode('utf-8')

    return dst_base64


def remove_bg(path) -> dict:
    # 画像読み込み
    img = cv2.imread(path)
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

            if 0.9 <= ellipse[1][0] / ellipse[1][1] <= 1.2 and 40 <= ellipse[1][1] <= 400:
                print(ellipse)
                # resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                if not isNanorInf(ellipse[0][0]) and not isNanorInf(ellipse[0][1]):
                    # 楕円描画
                    ellipses.append(
                        ellipse
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
    #
    # plt.imshow(img_a)
    # plt.show()

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img_a)

    return {'img': NdarrayToBase64(img_a), 'ellipses': ellipses}


if __name__ == '__main__':
    img = remove_bg('../images/panel1.jpg')

    plt.imshow(img['img'])
    plt.show()
