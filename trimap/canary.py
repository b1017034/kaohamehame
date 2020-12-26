import cv2
import numpy as np
import matplotlib.pyplot as plt


def crt():
    img = cv2.imread('../images/test2.jpg')

    img_blur_small = cv2.GaussianBlur(img, (15, 15), 0)
    plt.imshow(img_blur_small)
    plt.show()

    gray = cv2.cvtColor(img_blur_small, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()

    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh

    plt.imshow(thresh_im)
    plt.show()


    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(thresh_im, lowThresh, high_thresh)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    plt.imshow(edges, cmap='gray')
    plt.show()

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    plt.imshow(mask, cmap='gray')
    plt.show()


def isNanorInf(num: float) -> bool:
    return np.isnan(num) \
           and np.isnan(num) \
           and np.isinf(num) \
           and np.isinf(num)

def hough():
    img = cv2.imread('../images/test4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img, (15, 15), 0)

    # 二値化
    _, binimg = cv2.threshold(gray, 170, 255, cv2.THRESH_OTSU)
    binimg = cv2.bitwise_not(binimg)


    # 結果画像の黒の部分を灰色にする。
    bimg = binimg // 4 + 255 * 3 // 4
    resimg = cv2.merge((bimg, bimg, bimg))

    # 輪郭取得
    contours, hierarchy = cv2.findContours(binimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(binimg.shape)
    for i, cnt in enumerate(contours):
        # 楕円フィッティング
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)

            if 0.9 <= ellipse[1][0] / ellipse[1][1] <= 1.2 and 40 <= ellipse[1][1] <= 400:
                print(ellipse)
                resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                if not isNanorInf(ellipse[0][0]) and not isNanorInf(ellipse[0][1]):
                    cx = int(ellipse[0][0])
                    cy = int(ellipse[0][1])

                    # 楕円描画
                    mask = cv2.ellipse(mask, ellipse, (255, 255, 255), -1)


                    # img = img & mask

                    # cv2.drawMarker(resimg, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                    # cv2.putText(resimg, str(i + 1), (cx + 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 1, cv2.LINE_AA)

    mask_stack = np.dstack([mask]*3)

    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * (0.0, 0.0, 1.0))
    masked = (masked + 255).astype('uint8')

    c_blue, c_green, c_red = cv2.split(img)

    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    plt.imshow('mask', mask, cmap='gray')
    plt.show()

    plt.imshow(img_a)
    plt.show()

    cv2.imwrite("test4.png", mask)


if __name__ == '__main__':
    hough()
