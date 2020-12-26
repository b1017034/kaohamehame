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


def various_contours(path):
    color = cv2.imread(path)
    grayed = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayed, 218, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary)
    _, contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 90:
            continue

        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(color, c, -1, (0, 0, 255), 3)
        cv2.drawContours(color, [approx], -1, (0, 255, 0), 3)

    plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))


def hough():
    img = cv2.imread('../images/test2.jpg', cv2.IMREAD_COLOR)

    # グレイスケール化
    gray1 = cv2.bitwise_and(img[:, :, 0], img[:, :, 1])
    gray1 = cv2.bitwise_and(gray1, img[:, :, 2])

    # 二値化
    _, binimg = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binimg = cv2.bitwise_not(binimg)

    # 結果画像の黒の部分を灰色にする。
    bimg = binimg // 4 + 255 * 3 // 4
    resimg = cv2.merge((bimg, bimg, bimg))

    # 輪郭取得
    contours, hierarchy = cv2.findContours(binimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        # 楕円フィッティング
        ellipse = cv2.fitEllipse(cnt)
        print(ellipse)

        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])

        # 楕円描画
        resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
        cv2.drawMarker(resimg, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.putText(resimg, str(i + 1), (cx + 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 1, cv2.LINE_AA)

    cv2.imshow('resimg', resimg)
    cv2.waitKey()


if __name__ == '__main__':
    hough()
