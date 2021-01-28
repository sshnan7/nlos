import numpy as np
import imageio
import glob
import cv2

data_path = './vis/ver2_move'
data_path_list = glob.glob(data_path + '/*')
print("data list")
data_path_list = sorted(data_path_list)

start = 1000
number = 1000

img_arr = []

for i in range(start, start+number):
    img = cv2.imread(data_path_list[i])
    img_arr.append(img)

imageio.mimsave('./vis/demo_new_move_nlos.gif', img_arr, fps=10)

