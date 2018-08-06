from PIL import Image
import glob
import math
import cv2
import numpy as np
import os

# This class will generate starting, ending, and file path images

class Writter():
    def __init__(self):
        self.starting()
        print("created starting files")
        self.destination()
        print("created destination files")
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
        image_files = glob.glob('output/final/*.jpg')
        plane = cv2.imread("plane.png")
        plane = cv2.resize(plane,(100,100))

        iteration = 1
        for filename in path_files:
            image_file = image_files[iteration-1]
            print("path file ", filename," image ",image_file)
            
            image = cv2.imread(image_file)
            
            f = open(filename,'r')
            name = filename[5:len(filename)-4]
            
            # make directory for images
            directory = os.getcwd() + "\\output\\path\\" + name
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # make directory for ground truth
            directorygt = os.getcwd() + "\\output\\gt\\" 
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
                img = image.copy()
                x = pt[0]
                y = pt[1]

                if(0 >  x - plane.shape[1]//2 or 0 > y - plane.shape[0]//2 or img.shape[1] < x + plane.shape[1]//2 or img.shape[0] < y - plane.shape[0]//2):
                    continue
                
                img[y - plane.shape[0]//2 : y + plane.shape[0]//2, x - plane.shape[1]//2: x + plane.shape[1]//2] = plane
                img = cv2.resize(img,(512,512))
                
                #write images
                frameName = directory + "\\%d.jpg" % n
                n = n + 1

                # write a frame for each point
                cv2.imwrite(frameName, img)     
                
                b = plane.shape[1]
                h = plane.shape[0]
                coord = str(x)+"," +str(y)+"," +str(b)+"," +str(h) + "\n"
                
                #write that coordinate as ground truth
                t.write(coord)
                
            t.close()
            iteration = iteration + 1
        f.close()
    

    # Creates images from ending points
    def destination(self):
        print("Making ending files")
        start_files = glob.glob('end/*.txt')
        image_files = glob.glob('output/starting/*.jpg')
        iteration = 1
        for filename in start_files:
            print("end file ",filename)
            
            image = np.asarray(Image.open(image_files[iteration-1]))
            f = open(filename,'r')
            dest = [0,0]
            i = 0
            for line in f:
                dest[i] = self.discretize(line)
                i = i+1
            f.close()
            cv2.circle(image,(dest[0],dest[1]),10,(0,0,255),150)
            #image = cv2.resize(image, (512, 512)) 
            name = "final"
            self.write(image,filename[4:len(filename)-4],name)
            iteration = iteration + 1
        
    # Creates images from starting points
    def starting(self):
        print("Making starting files")
        start_files = glob.glob('start/*.txt')
        image = np.asarray(Image.open("MAP.png"))
        iteration = 1
        for filename in start_files:
            print("start file ",filename)
            
            f = open(filename,'r')
            dest = [0,0]
            i = 0
            for line in f:
                dest[i] = self.discretize(line)
                i = i+1
            f.close()
            img = image.copy()
            cv2.circle(img,(dest[0],dest[1]),10,(0,0,255),150)
            #img = cv2.resize(img, (512, 512)) 
            name = "starting"
            self.write(img,filename[6:len(filename)-4],name)
            iteration = iteration + 1
        
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
        base = float(line[:offset])
        val = int(base*math.pow(10,exp))
        return val

generate = Writter()