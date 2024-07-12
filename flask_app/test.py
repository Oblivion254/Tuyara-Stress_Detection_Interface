import cv2 
def video():
 cap = cv2.VideoCapture(0)
 success, frame = cap.read()
 cv2.imshow("frame", frame)
 cv2.waitkey(0)

video()