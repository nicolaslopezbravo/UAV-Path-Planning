# this script will compare where the algorithms pass through
from PIL import Image
import glob
import math
import cv2
import numpy as np
import os

class Metrics():
    def __init__(self,iteration,w):
        # Load map
        map_file = np.loadtxt('map.txt',dtype=int)
        MIN_VALUE = -10000

        # bounding negative values for keeping it in bounds
        map_file[0,:] = MIN_VALUE
        map_file[:,0] = MIN_VALUE
        map_file[:,len(map_file)-1]=MIN_VALUE
        map_file[len(map_file)-1,:]=MIN_VALUE
        
        Astar = self.drawAstar(iteration,map_file)
        DQN = self.drawDQN(iteration,map_file)
        SL = self.drawSL(iteration,map_file)
        
        a_length = sum(Astar)
        dqn_length = sum(DQN)
        sl_length = sum(SL)
        # this is making it into percentages
        for i in range(len(Astar)):
            Astar[i] /= a_length
            Astar[i] *= 100
            DQN[i] /= dqn_length
            DQN[i] *= 100
            SL[i] /= sl_length
            SL[i] *= 100
        # this is just formatting
        w.write(iteration+"\t") 
        w.write("%.2f\t" % Astar[0])
        w.write("%.2f\t" % Astar[1])
        w.write("%.2f\t" % Astar[2])
        w.write("%d\t\t" % a_length)
        w.write("%.2f\t" % DQN[0])
        w.write("%.2f\t" % DQN[1])
        w.write("%.2f\t" % DQN[2])
        w.write("%d\t\t" % dqn_length)
        w.write("%.2f\t" % SL[0])
        w.write("%.2f\t" % SL[1])
        w.write("%.2f\t" % SL[2])
        w.write("%d\n" % sl_length)
    # this method counts the amount of times the dqn is in an area
    def drawDQN(self,i,map_file):
        filename = str(int(i)-1)+"/" + i +".txt"
        f = open(filename,"r")
        points = []
        DQN = [0,0,0]
        for i in f:
            num = i.split(",")
            num[0] = int(num[0][1:])
            num[1] = int(num[1][1:len(num)-4])
            points.append(np.asarray(num))
        for pt in points:
            arr = pt
            x = arr[0]
            y = arr[1]
            if(map_file[x,y] == 0):
                DQN[1] += 1
            elif(map_file[x,y] == 1):
                DQN[0]+= 1
            else:
                DQN[2]+= 1
        return DQN
    # this method counts the amount of times the a straight line path is in each area
    def drawSL(self,i,map_file):
        filename = "SL_path/" + i +".txt"
        f = open(filename,"r")
        points = []
        SL = [0,0,0]
        for i in f:
            num = i.split(",")
            x = int(num[0])
            y = int(num[1])
            
            if(map_file[x,y] == 0):
                SL[1] += 1
            elif(map_file[x,y] == 1):
                SL[0]+= 1
            else:
                SL[2]+= 1
            
        return SL
    # this method counts the amount of times Astar is in an area
    def drawAstar(self,i,map_file):
        # we need i as input
        filename = "Astar_path/"+ i + ".txt"
        
        f = open(filename,'r')
        Astar = [0,0,0]
        dictionary = []
        for line in f:
            nums = line.split(',')
            new_size = [300,300]
            old_size = [7840,9695] #76008800
            x = int((int(nums[0])*new_size[0])/old_size[0])
            y = int((int(nums[1])*new_size[0])/old_size[1])
            if (x,y) in dictionary or x > 299 or y > 299:
                continue
            else:
                dictionary.append((x,y))
            if(map_file[x,y] == 0):
                Astar[1]+= 1
            elif(map_file[x,y] == 1):
                Astar[0]+= 1
            else:
                Astar[2]+= 1
                
        return Astar   
w = open("metrics.txt",'w+')
w.write("\t\t\t\tAstar\t\t\t\t\t\t\t\tDQNt\t\t\t\t\t\t\tStraightLine\n")
w.write("Case\tBuilding\tNeutral\tRoad\tTotal\tBuilding\tNeutral\tRoad\tTotal\tBuilding\tNeutral\tRoad\tTotal\n")
pts = [2,3,7,11,12,14]
for i in pts:
    metrics = Metrics(str(i),w)
w.close()
print("done")
