# 实验1 图像去燥、直方图均衡化(6%)
# 实验目的：学习实践图像处理基本操作
# 实验要求：编程实现对图像的基本去燥功能，及应用直方图均衡化实现对图像的增强处理。
# Author: HardyC-1021

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from exe3 import Car
# 使用中值滤波进行图像去噪,没考虑边界 src原图路径  dst新图路径 k核大小
def MedianFilter(src, dst, k, padding=None):
    imarray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    height, width = imarray.shape[:2]

    print(height,width)
    # print(imarray)
    if not padding:
        edge = int((k - 1) / 2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("K大的离谱了！")
            return None
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(height):
            for j in range(width):
                # 判断是否为边界
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= width - edge - 1:
                    new_arr[i, j] = imarray[i, j]
                else: # 中值滤波
                    new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
        new_im = Image.fromarray(new_arr)
        new_im.save(dst)

#图像直方图
def calcHist(src):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    img = img.flatten()
    wubbalubbadubdub = plt.hist(img, bins=256)  # 绘制直方图
    plt.show()
# 图像直方图均衡化
def hist_equal(src, dst):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    # 计算图像的灰度累计
    num = np.zeros(256,dtype="uint16")
    for i in range(0, h):
        for j in range(0, w):
            k = img[i, j]
            num[k] = num[k] + 1
    # 计算图像的灰度频率
    frequency = np.zeros(256)
    for i in range(0,255):
        frequency[i] = num[i] / (h * w)
    # print(frequency)
    # 积分累计函数
    acclu = np.cumsum(frequency)
    # print(acclu)
    # 四舍五入
    acclu = np.around(255 * acclu + 0.5)
    # print(acclu)
    # wubbalubbadubdub = plt.hist(acclu, bins=256)  # 绘制直方图
    # plt.show()
    new_img = np.zeros((h, w), dtype="uint8")
    # 把img对应new_img的值换入
    for i in range(0, h):
        for j in range(0, w):
            new_img[i, j] = acclu[img[i,j]]
    print(new_img)
    cv2.imwrite(dst, new_img)

src = "dog.jpeg"
dst = "IMG22.jpg"
k = 3
# MedianFilter(src, dst, k)
# calcHist(src)
hist_equal(src, dst)
