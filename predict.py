import os
import random
import tensorflow as tf
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from config import IMG_WIDTH, IMG_HEIGHT
import matplotlib.pyplot as plt

from cnn_model import cnn_model

# make GPU unavailable, only training needs to be on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# colorize images in folder
if __name__ == '__main__':
    TEST_DIR = './Dataset-Places2/test/'
    NUM_IMAGES = 10

    model = cnn_model()
    model.compile(optimizer='rmsprop', loss='mse')
    # Loads the weights
    model.load_weights('./Saved-weights/cnn-model-01.hdf5')

    color_me = []
    original = []
    for filename in random.sample(os.listdir(TEST_DIR), NUM_IMAGES):
        image = tf.keras.preprocessing.image.load_img(TEST_DIR + filename, color_mode='rgb',
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT))
        image = np.asarray(image)
        color_me.append(image)
        original.append(image)

    color_me = np.asarray(color_me)
    color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]  # only grayscale
    color_me = color_me.reshape(color_me.shape + (1,))

    # Test model
    output = model.predict(color_me)
    output = output * 128  # un-normalize

    for i in range(NUM_IMAGES):
        fig, axis = plt.subplots(1, 3, figsize=(25, 45))

        img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))  # zeroes matrix
        img[:, :, 0] = color_me[i][:, :, 0]  # grayscale
        img[:, :, 1:] = output[i]  # predicted two layers
        img = lab2rgb(img)

        input_img = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3))
        input_img[:, :, 0] = color_me[i][:, :, 0]
        input_img = lab2rgb(input_img)

        # grayscale, predicted, original
        axis[0].imshow(input_img)
        axis[0].axis('off')
        axis[0].set_title("Input Image", fontsize=15)
        axis[1].imshow(img)
        axis[1].axis('off')
        axis[1].set_title("Model Prediction", fontsize=15)
        axis[2].imshow(original[i])
        axis[2].axis('off')
        axis[2].set_title("Original Image", fontsize=15)

        plt.show()
