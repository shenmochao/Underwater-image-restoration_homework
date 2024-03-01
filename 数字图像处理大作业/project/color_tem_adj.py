import cv2
import numpy as np
import matplotlib.pyplot as plt

# 用法：你可以将neutral.png经过手机/PS处理得到新的比色板，然后作为map导入
# map的格式要求：.png 512*512像素
# 原理就是用map的颜色一一替换

def adjust_color_temperature(img, map, temperature):
    """
    对图片img进行滤镜map操作，并调整色温
    :param img: 原图片数组
    :param map: 滤镜色卡数组
    :param temperature: 色温调整值，可正可负
    :return: 调整色温后的图片数组
    """
    rows, cols = img.shape[:2]
    dst = np.zeros((rows, cols, 3), dtype="uint8")
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    r = r.astype(np.int16)
    g = g.astype(np.int16)
    b = b.astype(np.int16)

    x = (g // 4) + (b // 32) * 64
    y = (r // 4) + ((b % 32) // 4) * 64

    b_adjusted = b + temperature
    b_adjusted[b_adjusted < 0] = 0
    b_adjusted[b_adjusted > 255] = 255

    x_adjusted = (g // 4) + (b_adjusted // 32) * 64

    for i in range(rows):
        for j in range(cols):
            dst[i][j] = map[x_adjusted[i][j]][y[i][j]]

    return dst