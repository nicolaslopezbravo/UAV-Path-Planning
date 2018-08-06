import shutil
import os
filePath = input("Name of file to copy i.e '0/DQN.py': ")
from_files = int(input("starting range of files to copy it to i.e 1: "))
to_files = int(input("ending range of files to copy it to i.e 15: "))

for i in range(from_files,to_files+1):
    if i == int(filePath[:1]):
        continue
    folderPath = os.path.join(str(i), os.path.basename(filePath))
    shutil.copy(filePath, folderPath)
    
print("done")
