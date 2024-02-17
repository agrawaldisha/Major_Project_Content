import pickle
import os
import cv2
import face_recognition
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
imagebackground=cv2.imread("Resources/background.png")
FolderModePath='Resources/Modes'
modePathList=os.listdir(FolderModePath)
imageModeList=[]
for path in (modePathList):
    imageModeList.append(cv2.imread(os.path.join(FolderModePath,path)))
#Load the Encoding File
print("Loading Encodeing file")
file=open('EncodeFile.p','rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,studentsid=encodeListKnownWithIds
print("Loaded Encoded File")
while(True):
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)
    
    imagebackground[162:162+480,55:55+640]=img
    imagebackground[44:44+633,808:808+414]=imageModeList[3]

    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(matches)
        print(faceDis)
        
    cv2.imshow("Face Attendence",img)
    cv2.imshow("Face Attendence",imagebackground)
    
    cv2.waitKey(0)
