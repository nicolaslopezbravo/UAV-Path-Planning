
import numpy as np
import cv2
import glob
import math
#   this script takes the output generated from the matlab code and transforms it into what we need
#   matlab code has been modified to be used, however the matlab code is ridiculously large so it was not uploaded.
class findDestinations():
    def __init__(self,filename):
        starting_location_path = "starting_locations/"+filename+".txt"
        ending_location_path = "ending_locations/"+filename+".txt"
        self.start = self.transform(starting_location_path)
        self.dest = self.transform(ending_location_path)

    def transform(self,location):
        f = open(location,'r')
        dest = [0,0]
        i = 0
        for line in f:
            dest[i] = self.discretize(line)
            i = i+1
        f.close()
        new_size = [300,300]
        old_size = [7840,9695]
        x = int((dest[0]*new_size[0])/old_size[0])
        y = int((dest[1]*new_size[0])/old_size[1])
        return [x,y]
        

    # Discretizes from matlab output
    def discretize(self,line):
        offset = len(line)-5
        exp = int(line[offset+3:])
        base = float(line[:offset])
        val = int(base*math.pow(10,exp))
        return val
    
    def returnDestination(self):
        return self.dest
    
    def returnStarting(self):
        return self.start
f = open("test_destinations.txt","w+")
for i in range(1,116):
    finder = findDestinations(str(i))
    start = finder.returnStarting()
    end = finder.returnDestination()
    content = str(start[0])+','+str(start[1]) + ";"+ str(end[0])+',' + str(end[1]) + '\n'
    f.write(content)
f.close()

