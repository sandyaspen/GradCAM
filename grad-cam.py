from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

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
for file in range(1, 6):
    for rank in range(1, 4):
        img_path = "./data/gc" + str(file) + ".jpg"

        # Load image
        img = image.load_img(img_path, target_size=(224, 224))
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
        last_layer = model.get_layer('block5_conv3')
        heatmap_model = models.Model(
            [model.inputs], [last_layer.output, model.output])
        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model(x)
            loss = predictions[:, np.argsort(predictions[0], axis=None)[-rank]]
            grads = gtape.gradient(loss, conv_output)
            # akc:
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
        # calc LcGrad-CAM:
        heatmap = tf.reduce_mean(tf.multiply(
            pooled_grads, conv_output), axis=-1)
        # RELU
        heatmap = np.maximum(heatmap, 0)

        # Output
        heatmap = np.squeeze(heatmap)
        resized = cv2.resize(heatmap, (224, 224))
        normalized = None
        normalized = cv2.normalize(resized, normalized, alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormapped = cv2.applyColorMap(
            np.uint8(255 * (255 - normalized)), cv2.COLORMAP_JET)
        cv2img = cv2.imread(img_path)
        cv2img = cv2.resize(cv2img, (224, 224))
        final = cv2.addWeighted(colormapped, 0.6, cv2img, 0.4, 0.0)
        cv2.imwrite("./results/gc" + str(file) + "-Rank" +
                    str(rank) + "-IN.jpg", cv2img)
        cv2.imwrite("./results/gc" + str(file) +
                    "-Rank" + str(rank) + "-SI.jpg", final)
        cv2.imwrite("./results/gc" + str(file) + "-Rank" +
                    str(rank) + "-HM.jpg", colormapped)
