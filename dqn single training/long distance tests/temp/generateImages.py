from PIL import Image
import glob
import math
import cv2
import numpy as np
import os

# This class will generate starting, ending, and file path images

class Writter():
    def __init__(self):
        #self.starting()
        #print("created starting files")
        #self.destination()
        #print("created destination files")
        self.pathImages()
        print("created path files")
        print("done")       
        
    def get_line(self,start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
    
        >>> points1 = get_line((0, 0), (3, 4))
        >>> points2 = get_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
    
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

    def pathImages(self):
        print("Making path frames")

        # open and resize images, path points
        path_files = glob.glob('path/*.txt')
        
        for filename in path_files:
            print("path file ", filename)
            
            f = open(filename,'r')
            name = filename[5:len(filename)-4]
            
            # make directory for ground truth
            directorygt = os.getcwd() + "\\gt\\" 
            if not os.path.exists(directorygt):
                os.makedirs(directorygt)

            # format path points
            pts = []
            for line in f:
                ini = line.strip().split()
                x = self.discretize(ini[0]) 
                y = self.discretize(ini[1]) 
                pts.append((x,y))
            f.close()

            allpts = []
            newpts = []

            # append all points and points in between
            for i in range(1,len(pts)):
                allpts.append(pts[i-1])
                newpts = self.get_line(pts[i-1],pts[i])
                stride = 5
                curr = 0
                
                for p in newpts:
                    if(curr % stride == 0):
                        allpts.append(p)    
                    curr += 1
            allpts.append(pts[len(pts)-1])

            textfile = directorygt + "\\" + name + ".txt" 
            t = open(textfile,'w+')
            n = 1
            for pt in allpts:
                x = pt[0]
                y = pt[1]                
                b = 100
                h = 100
                coord = str(x)+"," +str(y)+"," +str(b)+"," +str(h) + "\n"
                
                #write that coordinate as ground truth
                t.write(coord)
                
            t.close()
            #iteration = iteration + 1
        f.close()
        
    # Writes images to specified file
    def write(self,img,n,name):
        directory = os.getcwd() + "\\output\\" + name 
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        frameName = directory + "\\" + n + ".jpg"
        cv2.imwrite(frameName, img)     # save frame as JPEG file

    # Discretizes from matlab output
    def discretize(self,line):
        offset = len(line)-5
        exp = int(line[offset+3:])
        print(exp)
        base = float(line[:offset])
        val = int(base*math.pow(10,exp))
        return val

generate = Writter()
