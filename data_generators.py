import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE


def create_train_generator():

    # Training ImagaDataGenerator with Augmentation
    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.2,
                                       rotation_range=15,  # 15 degrees
                                       horizontal_flip=True,
                                       data_format="channels_last",
                                       rescale=1.0 / 255.0,
                                       validation_split=0.7,  # for part of the data, change
                                       dtype=tf.float32,
                                       # preprocessing_function=chosen_preprocess
                                       )

    # Create a flow from the directory
    train_data_gen = train_datagen.flow_from_directory(directory='./Dataset-Places2/train',
                                                       subset='training',
                                                       shuffle=True,
                                                       seed=123,
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       class_mode=None
                                                       )
    return custom_generator(train_data_gen), train_data_gen.samples


def create_val_generator():

    # Validation ImageDataGenerator without Augmentation
    valid_datagen = ImageDataGenerator(data_format="channels_last",
                                       rescale=1.0 / 255.0,
                                       validation_split=0,
                                       dtype=tf.float32,
                                       # preprocessing_function=chosen_preprocess
                                       )

    # Create a flow from the directory for validation data, same seed
    val_data_gen = valid_datagen.flow_from_directory(directory='./Dataset-Places2/val',
                                                     subset='training',
                                                     shuffle=True,
                                                     seed=123,
                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode=None
                                                     )

    return custom_generator(val_data_gen), val_data_gen.samples

"""
def create_test_generator():
"""


# Customize generator
def custom_generator(gen):
    while True:
        batch = next(gen)
        lab_batch = rgb2lab(batch) # convert to lab
        X_batch = lab_batch[:, :, :, 0]  # grayscale layer
        Y_batch = lab_batch[:, :, :, 1:] / 128  # green-red and blue-yellow layers, normalize
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)
