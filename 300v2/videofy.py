import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

f = open("0.txt","r")
points = []
for i in f:
    num = i.split(",")
    for i in range(len(num)-1):
        num[i] = int(num[i][1:4])
    num[i+1] = int(num[i+1][1:len(num)-1])
    points.append(np.asarray(num))

mappy = cv2.imread("1.jpg")
plane = cv2.imread("plane.png")

mappy = cv2.resize(mappy,(300,300))

# each point will be an image
count = 0
for pt in points:
    arr = np.asarray(pt)
    x = int((arr[0] + arr[2]) // 2)
    y = int((arr[1] + arr[3]) // 2)
    plane = cv2.resize(plane,(10,10))
    img = mappy.copy()
    if(y >= 295):
        y = 294
    if(x >= 295):
        x = 294
    img[y - 5 : y + 5, x - 5: x + 5] = plane
    cv2.imwrite(os.getcwd()+"\\pt1\\frame%d.jpg" %count,img)
    count += 1
    print(count," out of ",len(points))
    #cv2.imshow('t',img)
    #cv2.waitKey(0)
