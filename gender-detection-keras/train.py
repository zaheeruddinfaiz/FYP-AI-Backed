# Author: Arun Ponnusamy
# website: http://www.arunponnusamy.com

# import necessary packages
import glob
import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model.smallervggnet import SmallerVGGNet
from sklearn.model_seleclction import train_test_split
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, default="gender_detection.model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = ap.parse_args()

# initial parameters
epochs = 1
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# load image files from the dataset
image_files = ["./combined/" +
               f for f in os.listdir("./combined") if not os.path.isdir(f)]
random.seed(42)
random.shuffle(image_files)

# create groud-truth label from the image path
for img in image_files:

    image = cv2.imread(img)

    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split('/')
    label = label[2].split('_')[0]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# build model
model = SmallerVGGNet.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save("./trained/mymodel.model")

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig(args.plot)
