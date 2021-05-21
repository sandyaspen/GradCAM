from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

# STEP 1:
model = VGG16(
include_top=True,
weights="imagenet",
input_tensor=None,
input_shape=None,
pooling=None,
classes=1000,
classifier_activation="softmax",
)

img_path = "./data/gc1.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# Run preprocessing
x = preprocess_input(x)

# STEP 2:
# Use the model to predict classes
prediction = model.predict(x)
# Display top 3 classes the model predicts
print("#"*40)
print("Top 3 classes predicted for the image:")
print(decode_predictions(prediction, top=3))

# STEP 3:
# Get the index of the top prediction
max_prediction = np.argmax(prediction, axis=1)[0]
print(prediction[0][max_prediction])
last_layer = model.get_layer('block5_conv3')
with tf.GradientTape() as gtape:

