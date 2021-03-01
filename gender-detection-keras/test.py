import cv2
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    cv2.imshow('Keras Faces', frame)
    if cv2.waitKey(5) == 27:  # ESC key press
        break
