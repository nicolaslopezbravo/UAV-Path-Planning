# this script will generate an image from A* and DQN and compare them
from PIL import Image
import glob
import math
import cv2
import numpy as np
import os

# This class will generate starting, ending, and file path images
class Compare():
    def __init__(self,iteration):
        self.drawAstar(iteration)
        self.drawDQN(iteration)
        print("done comparison")

    def drawDQN(self,i):
        filename = "../DQN_path/"+ i +".txt"
        f = open(filename,"r")
        image_file = '../results/' + i + '.jpg'
        image = cv2.imread(image_file)
        points = []
        
        for i in f:
            num = i.split(",")
            num[0] = int(num[0][1:])
            num[1] = int(num[1][1:len(num)-4])
            points.append(np.asarray(num))
        for pt in points:
            arr = pt
            x = arr[0]
            y = arr[1] 
            cv2.circle(image,(x,y),1,(255,0,255),-1)
            cv2.imwrite(image_file,image)
            
    # color is Yellow
    def drawAstar(self,i):
        # we need i as input
        # open and resize images, path points
        image_file = "../input_images/" + i + ".jpg" 
        filename = "../Astar_path/"+ i + ".txt"
        image = cv2.imread(image_file)
        
        f = open(filename,'r')
        
        for line in f:
            nums = line.split(',')
            new_size = [300,300]
            old_size = [7840,9695]
            x = int((int(nums[0])*new_size[0])/old_size[0])
            y = int((int(nums[1])*new_size[0])/old_size[1])
            
            cv2.circle(image,(x,y),1,(0,255,255),-1)
            cv2.imwrite(r'../results/' + i + '.jpg',image)
