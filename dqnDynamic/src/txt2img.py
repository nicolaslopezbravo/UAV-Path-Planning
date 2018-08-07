import numpy as np
from PIL import Image
import cv2
MIN_VALUE = -100000

map_file = np.loadtxt('map.txt',dtype=int)
image_file = np.zeros_like(cv2.imread('1.jpg'))

# bounding negative values for keeping it in bounds
map_file[0,:] = MIN_VALUE
map_file[:,0] = MIN_VALUE
map_file[:,len(map_file)-1]=MIN_VALUE
map_file[len(map_file)-1,:]=MIN_VALUE

# this script creates a different version of the map
for i in range(300):
    for j in range(300):
        if map_file[i,j]== 0:
            image_file[i,j] = [0,0,0] #nothing
        elif map_file[i,j] == 1:
            image_file[i,j] = [0,255,0] #green
        else:
            image_file[i,j] = [0,0,255] #red

cv2.imwrite('dqn_eyes_color_corrected.jpg',image_file)
