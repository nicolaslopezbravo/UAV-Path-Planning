import numpy as np
import cv2
from cv2 import VideoWriter
import os

class make_frames():
    def __init__(self,filename,iteration):
        f = open(filename,"r")
        points = []
        for i in f:
            num = i.split(",")
            num[0] = int(num[0][1:])
            num[1] = int(num[1][1:len(num)-4])
            points.append(np.asarray(num))

        mappy = cv2.imread("mappy.jpg")
        plane = cv2.imread("plane.png")

        mappy = cv2.resize(mappy,(300,300))

        # each point will be an image
        # here we write the frames, but instead lets make the video

        video_name = str(iteration)+'_results.avi'

        height, width, layers = mappy.shape
        fourcc = cv2.cv.CV_FOURCC(*'XVID') #cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_name, fourcc, 20.0, (width,height))

        for pt in points:
            arr = np.asarray(pt)
            y = arr[0]
            x = arr[1] 
            plane = cv2.resize(plane,(10,10))
            img = mappy.copy()
            if(y >= 295):
                y = 294
            if(x >= 295):
                x = 294
	    if(x <= 5):
		x = 6
	    if(y <= 5):
		y = 6
            img[y - 5 : y + 5, x - 5: x + 5] = plane
            video.write(img)
        
        video.release()
        cv2.destroyAllWindows()
    


