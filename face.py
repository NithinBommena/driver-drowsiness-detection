import cv2
import sys
import os

currentframe=0

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')

cap= cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
try:
   if not os.path.exists('data'):
       os.makedirs('data')
except OSError:
   print('Error: Creating directory of data')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        cv2.putText(frame,"Face Detected",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if ret:
        name='./data/frame'+ str(currentframe) + '.jpg'
        cv2.imwrite(name,frame)
        currentframe += 1
    else:
        break
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()