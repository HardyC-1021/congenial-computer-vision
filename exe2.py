# 实验目的：实现边缘检测、Hough变换
# 实验要求：自选一种边缘检测算子实现对图像的边缘检测；在边缘检测的基础上，应用Hough变换算法提取最可能存在于图像上的3条直线，给出直线表达式。
# Canny边缘检测
# Author: HardyC-1021
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


def Canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # 用高斯滤波处理原图像降噪
    canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值

    sigma1 = sigma2 = 1.5
    sum = 0

    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)  # 生成二维高斯分布矩阵
                                                + (np.square(j - 3) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum

    # step1.高斯滤波
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    W, H = gray.shape
    print(W,H)
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")


    # step2.增强 通过求梯度幅值
    W1, H1 = new_gray.shape
    print(W1,H1)
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值


    dx= np.maximum(dx, 1e-10)
    angle = np.arctan(dy / dx)
    angle = angle / np.pi * 180

    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype=np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

    # setp3.非极大值抑制 NMS

    H, W = angle.shape
    _edge = np.clip(d, 0, 255).copy()
    for y in range(H):
        for x in range(W):
            dx1, dy1, dx2, dy2 = 1, 1, 1, 1
            if _angle[y, x] == 0:
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            elif _angle[y, x] == 45:
                dx1, dy1, dx2, dy2 = -1, 1, 1, -1
            elif _angle[y, x] == 90:
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            elif _angle[y, x] == 135:
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1
            # 边界处理
            if x == 0:
                dx1 = max(dx1, 0)
                dx2 = max(dx2, 0)
            if x == W - 1:
                dx1 = min(dx1, 0)
                dx2 = min(dx2, 0)
            if y == 0:
                dy1 = max(dy1, 0)
                dy2 = max(dy2, 0)
            if y == H - 1:
                dy1 = min(dy1, 0)
                dy2 = min(dy2, 0)
            if max(max(d[y, x], d[y + dy1, x + dx1]), d[y + dy2, x + dx2]) != d[y, x]:
                _edge[y, x] = 0

    # step4. 双阈值算法检测、连接边缘
    W3, H3 = _edge.shape
    DT = np.zeros([W3, H3])
    # 定义高低阈值
    TL = 0.2 * np.max(_edge)
    TH = 0.3 * np.max(_edge)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (_edge[i, j] < TL):
                DT[i, j] = 0
            elif (_edge[i, j] > TH):
                DT[i, j] = 1
            elif ((_edge[i - 1, j - 1:j + 1] < TH).any() or (_edge[i + 1, j - 1:j + 1]).any()
                  or (_edge[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    return DT,canny

def Hough(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 300)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0))
        a, b, c = LINE(x1,x2,y1,y2)
        print('%.2fx + %.2fy + %.2f = 0' %(a,b,c))
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img[:, :, ::-1]), plt.title('Hougu')
    plt.xticks([]), plt.yticks([])
    plt.show()
def LINE(x1,x2,y1,y2):
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    return a, b, c

if __name__ == "__main__":
    img = cv2.imread('dilireba.png')
    print(img.shape)
    DT, canny = Canny(img)
    Hough(img)
    # plt.figure(1)
    # 第一行第一列图形
    # ax1 = plt.subplot(1, 3, 1)
    # plt.sca(ax1)
    # plt.imshow(img)
    # plt.title("artwork")
    # # 第一行第二列图形
    # ax2 = plt.subplot(1, 3, 2)
    # plt.sca(ax2)
    # plt.imshow(canny, cmap="gray")
    # plt.title("opencv Canny")
    #
    # ax3 = plt.subplot(1, 3, 3)
    # plt.sca(ax3)
    # plt.imshow(DT, cmap="gray")
    # plt.title("my Canny")
    # plt.show()
