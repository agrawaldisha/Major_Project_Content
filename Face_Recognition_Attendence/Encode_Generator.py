import cv2
import face_recognition
import pickle
import os 
FolderPath='Images'
PathList=os.listdir(FolderPath)
imgList=[]
studentsid=[]
for path in (PathList):
    imgList.append(cv2.imread(os.path.join(FolderPath,path)))
    studentsid.append(os.path.splitext(path)[0])
print(len(imgList))
print(studentsid)


def findEncodings(imagesList):
    encodeList=[]
    for image in imagesList:
        img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
        return encodeList
print("Encoded Started .....")
encodeListKnown=findEncodings(imgList)
print("Encoding Ended")
encodeListKnownWithIds=[encodeListKnown,studentsid]
file=open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")
