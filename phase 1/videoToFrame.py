import cv2
import glob
import os
basewidth = 256
videos = glob.glob('videos/*.avi')

for vid in videos:
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        directory = os.getcwd() + "\\" + vid[:len(vid)-4]
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        frameName = directory + "\\frame%d.jpg" % count
        cv2.imwrite(frameName, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1