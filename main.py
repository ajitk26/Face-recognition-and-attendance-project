import cv2
import numpy as np
import face_recognition

elonImg=face_recognition.load_image_file('imageSet/Elon Musk.jpg')
elonImg=cv2.cvtColor(elonImg,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('imageSet/Elon test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(elonImg)[0]
encodeElon=face_recognition.face_encodings(elonImg)[0]
cv2.rectangle(elonImg,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeimgTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0],faceLocTest[1],faceLocTest[2]),(255,0,255),2)
result=face_recognition.compare_faces([encodeElon],encodeimgTest)
faceDis=face_recognition.face_distance([encodeElon],encodeimgTest)
cv2.putText(imgTest,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
print(result,faceDis)


cv2.imshow('Elon Musk',elonImg)
cv2.imshow('Elon test',imgTest)
cv2.waitKey(0)


