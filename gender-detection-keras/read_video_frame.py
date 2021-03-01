import cv2
cap = cv2.VideoCapture(0)
count = 0
import time
first = True

def skip_frames(seconds, cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    for i in range(seconds * int(fps)):
        cap.read()

def get_frame(seconds):
    ret, frame = cap.read()
    skip_frames(seconds,cap)
    # cv2.imshow('window-name', frame)
    # # cv2.imwrite("frame%d.jpg" % count, frame)
    # count = count + 1
    # if not first:
    #     time.sleep(10)
    # else:
    #     first = False    
    return frame
