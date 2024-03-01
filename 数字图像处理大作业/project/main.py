import os
import cv2
import natsort
import numpy as np
import white_balance
import clahe
import color_tem_adj


def Underwater_Image_Restoration(image, map, temperature):
    # 根据图片颜色是否丰富,选择白平衡算法
    if white_balance.image_colorfulness(image) < 30.0:
        image_white_balance = white_balance.Perfect_reflection(image)
    else:
        image_white_balance = white_balance.gray_world(image)
    # 直方图均衡化
    image_clahe = clahe.clahe(image_white_balance)
    # 调整色温
    image_end = color_tem_adj.adjust_color_temperature(image_clahe, map, temperature)
    return image_end


np.seterr(over="ignore")
if __name__ == "__main__":
    pass
folder = "C:/Users/Summer/Desktop/G3/MID/PRO_F/project"
# 注意，路径中不能有中文
path = folder + "/InputImages"
files = os.listdir(path)
files = natsort.natsorted(files)

map = cv2.imread(folder + "/Others/" + "64.png")
temperature = 0

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split(".")[0]
    if os.path.isfile(filepath):
        print("********    file   ********", file)
        img = cv2.imread(folder + "/InputImages/" + file)
        end_image = Underwater_Image_Restoration(img, map, temperature)
        cv2.imwrite("OutputImages/" + prefix + "_end.jpg", end_image)
