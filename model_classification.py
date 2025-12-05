import os

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import pandas as pd
import os

model_path = "my_model.keras"
image_size = (224, 224)
classes = sorted(os.listdir("bilder"))

model = keras.models.load_model(model_path)

img = keras.utils.load_img("input3.jpeg", target_size=image_size)
#plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)[0]
print(predictions)
print(classes)

for i in range(5):
    print(f"{classes[i]} - {predictions[i] * 100:.10f}%")
#score = float(keras.ops.sigmoid(predictions[0][0]))
#print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
