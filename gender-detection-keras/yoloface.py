import argparse
import sys
import os
import cv2
from utils import *
from detect_gender_webcam import *
import numpy as np
from realtime_demo import FaceCV
# check outputs directory
args = {
    "model_cfg": "./cfg/yolov3-face.cfg",
    "model_weights": "./model-weights/yolov3-wider_16000.weights",
    "src": 0
}

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args['model_cfg'], args['model_weights'])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# cap = cv2.VideoCapture(args['src'])
wind_name = 'face detection using YOLOv3'
# cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
age_predictor = FaceCV()


def get_age_gender(frame):

    # Get data from the camera

    # while True:
    predicted_gender = 0
    predicted_age = 0

    # has_frame, frame = cap.read()
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    # initialize the set of information we'll displaying on the frame
    info = [
        ('number of faces detected', '{}'.format(len(faces)))
    ]
    # print(info)
    for face in (faces):
        # print(face)
        startX = np.dtype('int64').type(face[0])
        startY = np.dtype('int64').type(face[1])
        endX = np.dtype('int64').type(face[0]+face[2])
        endY = np.dtype('int64').type(face[1]+face[3])
        face_crop = np.copy(frame[startY:endY, startX:endX])
        predicted_gender = get_gender(face, face_crop, frame)
        predicted_age = age_predictor.detect_face(face, frame)
        predicted_age = int(predicted_age)
        cv2.rectangle(frame, (face[0], face[1]),
                      (face[0]+face[2], face[1]+face[3]), (255, 0, 0))
        cv2.putText(frame, f"{predicted_gender} {predicted_age}", (startX, startY),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # if cv2.waitKey(5) == 27:  # ESC key press
    #     return
    try:
        cv2.imshow('FYP', frame)
        cv2.waitKey(1)
        if(predicted_age != 0 and len(predicted_gender) != 0):
            return [predicted_age, predicted_gender]
        else:
            return None
    except TypeError:
        return None
