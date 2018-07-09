from PIL import Image
import glob
import math
import cv2
import numpy as np
import os
# THIS FILE IS NO LONGER NEEDED
class GroundTruth():
    def __init__(self):
        self.pathWritter()
        print("done")
    
    # Discretizes from matlab output
    def discretize(self,line):
        offset = len(line)-5
        exp = int(line[offset+3:])
        base = float(line[:offset])
        val = int(base*math.pow(10,exp))
        return val
        
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

    def pathWritter(self):
        path_files = glob.glob('path/*.txt')
        image_files = glob.glob('output/final/*.jpg')
        iteration = 1
        plane = cv2.imread("plane.png")
        plane = cv2.resize(plane,(100,100))

        for filename in path_files:
            print("path file ", iteration)
            image = cv2.imread(image_files[iteration-1])
            f = open(filename,'r')

            name = str(iteration)
            directory = os.getcwd() + "\\output\\gt\\" + name
            if not os.path.exists(directory):
                os.makedirs(directory)

            pts = []

            for line in f:
                ini = line.strip().split()
                x = self.discretize(ini[0]) 
                y = self.discretize(ini[1]) 
                pts.append((x,y))
            allpts = []
                       
            for i in range(1,len(pts)):
                allpts.append(pts[i-1])
                # call new pts with i-1 and i
                newpts = self.get_line(pts[i-1],pts[i])
                stride = 5
                curr = 0
                for p in newpts:
                    if(curr % stride == 0):
                        allpts.append(p)    
                    curr += 1
                allpts.append(pts[i])
            
            textfile = directory + "\\%d.txt" % iteration
            t = open(textfile,'w+')
            
            for pt in allpts:
                img = image.copy()
                x = pt[0]
                y = pt[1]

                if(0 >  x - plane.shape[1]//2 or 0 > y - plane.shape[0]//2 or img.shape[1] < x + plane.shape[1]//2 or img.shape[0] < y - plane.shape[0]//2):
                    continue
                b = plane.shape[1]
                h = plane.shape[0]
                coord = str(x)+"," +str(y)+"," +str(b)+"," +str(h) + "\n"
                t.write(coord)
            t.close()
            iteration = iteration + 1
        f.close()

gt = GroundTruth()