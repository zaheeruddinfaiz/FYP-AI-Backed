from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv

# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                      cache_subdir="pre-trained", cache_dir=os.getcwd())

# load model
model = load_model(model_path)

classes = ['man', 'woman']

# webcam = cv2.VideoCapture(0)

# while webcam.isOpened():
#     status, frame = webcam.read()

#     # apply face detection
#     face, confidence = cv.detect_face(frame)

    

def get_gender(face_coordinates, face_crop, frame):
    (startX, startY) = face_coordinates[0], face_coordinates[1]
    (endX, endY) = face_coordinates[2], face_coordinates[3]

    # draw rectangle over face
    # cv2.rectangle(frame, (startX+5, startY+5), (endX, endY), (0, 255, 0), 2)

    # crop the detected face region
    # face_crop = np.copy(frame[startY:endY, startX:endX])

    if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
        return None

    # preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max accuracy
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = startY - 10 if startY - 10 > 10 else startY + 10

    # write label and confidence above face rectangle
    # cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                # 0.7, (0, 255, 0), 2)

    # display output
    # cv2.imshow("gender detection", frame)
    return label
