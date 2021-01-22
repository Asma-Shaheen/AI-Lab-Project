import cv2#import from laibrary openCV

cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')#instatiation and Adding harcascade classifire for focedetector by providing location of script
cascade_classifier1 = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')#instatiation and Adding harcascade classifire for Eyedetector by providing location of script

VCapture = cv2.VideoCapture(0)#instansiating the object from videocapture class, where 0 means default video source

while True:#for continous deduction
    
    ret, frame = VCapture.read()# Captureing frame-by-frame
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)#conveting 3channel RGB Image to gray scale image
    if(len(detections) > 0):#deduct all possible regions in the environment where posiblity of presence of face 
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        for (x,y,w,h) in detections:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
    detections1 = cascade_classifier1.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)#conveting 3channel RGB Image to gray scale image
    if(len(detections1) > 0):#deduct all possible regions in the environment where posiblity of presence of face 
        (x,y,w,h) = detections1[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        for (x,y,w,h) in detections:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow('AI_DETECTOR',frame)#displaying frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

VCapture.release()# Closes video file or capturing device.
cv2.destroyAllWindows()#destroys all the windows we created
