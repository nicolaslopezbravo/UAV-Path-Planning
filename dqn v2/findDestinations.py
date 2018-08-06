import numpy as np
import cv2
import glob
import math
class findDestinations():
    def __init__(self,filename):
        locations = "destinations/"+filename+".txt"
        #locations = filename
        self.transform(locations)

    def transform(self,locations):
        f = open(locations,'r')
        line = f.read()
        nums = line.split(';')
        start = nums[0].split(',')
        end = nums[1].split(',')
        self.start = [0,0]
        self.dest = [0,0]
        self.start[0] = int(start[0])
        self.start[1] = int(start[1])
        self.dest[0] = int(end[0])
        self.dest[1] = int(end[1])
        
    def returnDestination(self):
        return self.dest
    
    def returnStarting(self):
        return self.start
