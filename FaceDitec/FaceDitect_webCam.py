## Letd do that for videos
import cv2

#Load some pre-trained data on fontalface XML
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose a video to detect faces
webCam = cv2.VideoCapture(0)

#run every video frame one by one
while True:
    sucessfull_frame_read, frame = webCam.read()

    #convert to gray scale
    grayScale_webCam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayScale_webCam)

    #draw rectangle around the face
    for(x, y, w, h) in face_coordinates:
        #cv2.rectangle(imageSource, topLeftCoordinates, (topLeft+rectangel with and height), color, thikness)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 256, 0), 5)

    #Show webcam in custom window   
    cv2.imshow('Face Ditect Web Cam', frame)

    #program will run untill click any button
    key = cv2.waitKey(1)

    #exit from loop when we press Q or q
    if key==81 or key==113:
        break

#release resources
webCam.release()
