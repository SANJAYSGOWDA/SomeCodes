# this checks the image created time not the modified if you wnat modified use os.path.getmtime(path)


import os
from datetime import datetime
from tqdm import tqdm

getClassandImage = {}

def getFileCreationDate(path):
    timeStamp = os.path.getctime(path)
    timed = datetime.fromtimestamp(timeStamp)
    return timed

def bulildClassImageDict(folderPath, timeandDateCondition):
    for subFolder in tqdm(os.listdir(folderPath)):
        subFolderPath = folderPath + "\\" + subFolder
        imagePathList =[]
        for imgs in tqdm(os.listdir(subFolderPath), desc=f"{subFolder}"):
            if imgs.endswith('jpg'):
                imgPath = subFolderPath + "\\" + imgs
                creationTime = getFileCreationDate(imgPath)
                if(creationTime >= timeandDateCondition):
                    # print(creationTime)
                    imagePathList.append(imgPath)
        getClassandImage[subFolder] = imagePathList
    return getClassandImage

folderPath =r"\\10.15.1.24\Video1\08.SOFTWARE DEPARTMENT\Bhavani\Catalonia_Images\15April2025\Images"

timeandDateCondition = datetime(2025, 4, 18, 6, 30, 0 )
getClassandImage = bulildClassImageDict(folderPath, timeandDateCondition)


with open(r"D:\saveimage.txt", "w") as f:
    f.write(str(getClassandImage))
