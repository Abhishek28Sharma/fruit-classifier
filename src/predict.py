import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to test image")
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model("../fruit_classifier.h5")

# Preprocess image
img = image.load_img(args.image, target_size=(100,100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_idx = np.argmax(prediction)

# Map classes
class_labels = ["apple", "banana", "orange"]  # adjust based on dataset
print("Predicted:", class_labels[class_idx])
