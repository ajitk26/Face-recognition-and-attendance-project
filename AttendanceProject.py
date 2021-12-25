import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path="imgAttendance"
images=[]
clsNames=[]
myLst=os.listdir(path)
print(myLst)
for x in myLst:
    xImg=cv2.imread(f'{path}/{x}')
    images.append(xImg)
    clsNames.append(os.path.splitext(x)[0])
print(clsNames)

def getEncodings(images):
    encodeLst=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeLst.append(encode)
    return encodeLst

def markAttendance(name):
    with open('Attendance.csv', 'r+') as y:
        dataList= y.readlines()
        nameList=[]
        for line in dataList:
            entries=line.split(',')
            nameList.append(entries[0])
        if name not in nameList:
            now=datetime.now()
            dateString=now.strftime('%H:%M:%S')
            y.writelines((f'\n{name},{dateString}'))

markAttendance(('Golu'))


encodeLstKnown=getEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    complete,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    curFrameFaces = face_recognition.face_locations(imgS)
    curFrameEncoding = face_recognition.face_encodings(imgS, curFrameFaces)

    for encodeFace,faceLoc in zip(curFrameEncoding,curFrameFaces):
        matches=face_recognition.compare_faces(encodeLstKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeLstKnown,encodeFace)
        print(faceDis)
        matchesIndex=np.argmin(faceDis)

        if matches[matchesIndex]:
            name=clsNames[matchesIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)






    cv2.imshow('Webcam', img)
    cv2.waitKey(1)






