import cv2
import argparse
import os
import re
import sys
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def writeVideos(image_folder,ext):
    video_name = image_folder + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith("."+ext)]
    images = sorted_aphanumeric(images)


    frame = cv2.imread(os.getcwd() + "\\"+image_folder+"\\"+images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 20.0, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    cv2.destroyAllWindows()

ext = "jpg"
folder = "pt1"
writeVideos(folder,ext)
print("video " + folder + " written")
print("done")
input("press any key to continue..")
