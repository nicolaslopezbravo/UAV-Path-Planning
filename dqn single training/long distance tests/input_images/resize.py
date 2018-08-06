import glob
import cv2
import numpy as np
import re
import cv2
import glob
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

images = glob.glob("*.jpg")
images = sorted_aphanumeric(images)
count = 1
for i in images:
    img = cv2.imread(i)
    img = cv2.resize(img,(300,300))
    cv2.imwrite(str(count) + ".jpg",img)
    count += 1    

