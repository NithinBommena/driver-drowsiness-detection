import numpy as np
import cv2


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye =  cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while 1:
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes =leye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,100),1)
        eyes2=reye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes2:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,100),1)
        cv2.putText(frame,"Eyes Detected",(10,height-20),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()