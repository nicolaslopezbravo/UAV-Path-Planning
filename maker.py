import cv2
import numpy as np 
import PIL
from PIL import Image
roads = cv2.imread("maybego.png")[:,:,0]
buildings = cv2.imread("go.png")[:,:,0]
arr = np.zeros_like(roads)

for i in range(roads.shape[0]):
    for j in range(buildings.shape[1]):
        if(roads[i,j] < 100):
            arr[i,j] += -1
        if(buildings[i,j] > 100):
            arr[i,j] += 1
np.savetxt('map.txt',arr,fmt="%d")
#arr += 50
#final = Image.fromarray(arr)
#final.save('test.png')
