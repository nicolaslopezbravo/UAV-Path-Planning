import numpy as np
import cv2

img = np.asarray(cv2.imread("mappy.jpg"))

median = []
color = [200,30,30] # red
color = [30,30,200] # blue
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(img[i,j,2] >= color[2] and img[i,j,1] <= color[1] and img[i,j,0] <= color[0]):
            median.append((i,j))
x = 0
y = 0
for tp in median:
    x += tp[0]
    y += tp[1]
if len(median) < 1:
    median.append(0)
x = int(x / len(median))
y = int(y / len(median))

dest = (x,y)
print(dest)