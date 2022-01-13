import cv2
import os
import glob
import numpy as np

import tensorflow as tf

image_dir = "/home/karlis/Desktop/dataset/test/image"

model = tf.keras.models.load_model("/home/karlis/Documents/circular-padding/logs1/best_loss.h5")

for image in glob.glob(os.path.join(image_dir, "*")):
    image = cv2.imread(image)
    image = cv2.resize(image, (512, 256))

    tensor = image / 255.0
    tensor = tensor[np.newaxis, ...]

    mask = model.predict(tensor)
    mask = mask[0]
    mask = mask * 255

    cv2.imshow("mask", mask)
    cv2.imshow("img", image)
    cv2.waitKey(0)
