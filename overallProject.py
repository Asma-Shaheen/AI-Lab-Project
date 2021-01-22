import cv2
import numpy as np
import face_recognition as fr

def face_Detect():

    cascade_classifier = cv2.CascadeClassifier('MyData/haarcascade_frontalface_default.xml')#instatiation and Adding harcascade classifire for focedetector by providing location of script
    cascade_classifier1 = cv2.CascadeClassifier('MyData/haarcascade_eye.xml')#instatiation and Adding harcascade classifire for Eyedetector by providing location of script

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

def face_Recognize():
    webcam = cv2.VideoCapture(0) #capture video from webcam

    image1 = fr.load_image_file("knownFaces/PrinceWilliam.jpg") #load image 1
    image2 = fr.load_image_file("knownFaces/KateMiddleton.jpg") #load image 2
    analyzeFace1 = fr.face_encodings(image1)[0] #analyze image 1
    analyzeFace2 = fr.face_encodings(image2)[0] #analyze image 2

    recognized_Faces = [analyzeFace1, analyzeFace2] #recognize faces from given images
    recognized_Names = ["William", "Kate"] #recognize names from given images

    while True:
        ret, frame = webcam.read() #add frame to face to read it from webcam
        frameColour = frame[:, :, ::-1] #colour of frame to RBG

        locate_Faces = fr.face_locations(frameColour) #locate faces with frame
        face_Analyze = fr.face_encodings(frameColour, locate_Faces) #analyze faces from webcam

        for (top, right, bottom, left), face_encoding in zip(locate_Faces, face_Analyze): #match located faces with their analysis

            face_Match = fr.compare_faces(recognized_Faces, face_encoding) #match webcam faces with loaded images
            face_label = "Unknown" #if no face recognized
            distances_Faces = fr.face_distance(recognized_Faces, face_encoding) #distance checking for similarity between image face and webcam face 

            best_FaceMatch = np.argmin(distances_Faces) #find best face match
            if face_Match[best_FaceMatch]: 
                face_label = recognized_Names[best_FaceMatch] #if minimum distance found between compared faces, recognize face with their names
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) #creating rectangular border around face
            cv2.rectangle(frame, (left, bottom -20), (right, bottom), (0, 0, 255), cv2.FILLED) #lower rectangle border for face label
            label_Font = cv2.FONT_HERSHEY_DUPLEX #font for face label
            cv2.putText(frame, face_label, (left + 6, bottom - 6), label_Font, 1.0, (255, 255, 255), 1) #add label to the recognized face

        cv2.imshow('AI_FACE_RECOGNITION', frame) #display webcam window

        if cv2.waitKey(1) & 0xFF == ord('e'): #enter 'e' to exit
            break

    webcam.release() #release webcam video
    cv2.destroyAllWindows() #destroy opened windows afrer exiting

print("\nWELCOME TO FACE DETECTION + RECOGNITION\n")
print("[1] FACE DETECTION\n[2] FACE RECOGNITION")
value = input("\nYOUR INPUT = ")
if(value== "1"):
    face_Detect()
elif(value=="2"):
    face_Recognize()
else:
    print("INVALID INPUT!")


