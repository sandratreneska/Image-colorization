import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_WIDTH, IMG_HEIGHT
from skimage.color import rgb2lab, lab2rgb

# Paths
IMG_PATH = sys.argv[1]
GEN_PATH = sys.argv[2]
SAVE_PATH = sys.argv[3]

print(IMG_PATH)
print(GEN_PATH)

# Read image
image = tf.keras.preprocessing.image.load_img(IMG_PATH, color_mode='rgb', target_size=(IMG_HEIGHT, IMG_WIDTH))

# Preprocess
labImg = rgb2lab(image)
L = labImg[:, :, 0]
L_scaled = labImg[:, :, 0] / 50. - 1

# Preprocess for pre-trained model
L_scaledS_tacked = tf.stack((tf.reshape(L_scaled, (-1, 256, 256)), tf.reshape(L_scaled, (-1, 256, 256)), tf.reshape(L_scaled, (-1, 256, 256))), axis=3)

# Pre-trained model
g_model = keras.models.load_model(GEN_PATH)
colorScaled = g_model.predict(L_scaledS_tacked)

# Create final colored image
img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
print(img.shape)
img[:, :, 0] = L  # grayscale
img[:, :, 1:] = colorScaled * 128.  # predicted two layers un-normalized
img = lab2rgb(img)
print(img.shape)

# Save image
plt.imsave(SAVE_PATH + IMG_PATH.split('/')[-1], img)