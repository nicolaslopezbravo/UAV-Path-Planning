import glob
import cv2
import numpy as np
from findDestinations import findDestinations
# this is a small test to make sure the destinations are correct. it works with the old version of findDestinations and makeDestinationList
for i in range(1,5):
    img = str(i)
    print("starting iteration", img)
    image = np.asarray(cv2.imread("input_images/"+img+'.jpg'))
    finder = findDestinations(img)
    end = finder.returnDestination()
    start = finder.returnStarting()
    cv2.circle(image,(start[0],start[1]),10,(0,255,255),0)
    cv2.circle(image,(end[0],end[1]),10,(0,255,255),0)
    cv2.imwrite(r'test_images'+img+'.jpg',image)
    
    

