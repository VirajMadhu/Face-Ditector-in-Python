import cv2

from random import randrange #to obtain random range

#Load some pre-trained data on fontalface XML
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces
img = cv2.imread('RDJ.png')

#convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
 
#draw rectangle around the face
for(x, y, w, h) in face_coordinates:
    #cv2.rectangle(imageSource, topLeftCoordinates, (topLeft+rectangel with and height), color, thikness)
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 5)


#show image in our custome window
cv2.imshow('Face Ditector', img)

#program will run untill click any button
cv2.waitKey()

#release resources
img.release()