#imports the OpenCV library, which is a collection of functions and tools for computer vision
import cv2
#The PoseDetector class from cvzone.PoseModule helps with detecting human poses using the Mediapipe library.
from cvzone.PoseModule import PoseDetector

#This object will be used to detect human poses in images.
detector=PoseDetector()
#video capture helps to access the webcam and (0) means we are accessing the first webcam in our system
cap=cv2.VideoCapture(0)
#we set our captured video frame size to 3x4 widthxheight and 480x640 pixels 
cap.set(3,640)
cap.set(4,480)

while True:
    success,img=cap.read()
#It returns the processed image with pose landmarks drawn on it
    img=detector.findPose(img)
#imlist: a list containing the positions of detected keypoints (e.g., joints, limbs) 
#bbox: The bounding box coordinates that enclose the detected pose.
    imlist,bbox=detector.findPosition(img)
#it will print the elements in array if there is any human detection
#otherwise it will print an empty array to help keeping tracks/logs in form of array
    print(imlist)
    if len(imlist)>0:
        print('Human Detected')

# displays the captured frame in a window named 'Output'.
    cv2.imshow('Output',img)
#the wait shows the delay in capturing all the images and combining it into one video like frame with a wait of 1 milli second
    cv2.waitKey(1)
#to stop the capturing of image we we'll click "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
#it will help to terminate the running output window of camera monitoring

cap.release()
cv2.destroyAllWindows()
