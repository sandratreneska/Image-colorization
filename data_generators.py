import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, TRAIN_DIR, VAL_DIR


def create_train_generator():

    # Training ImagaDataGenerator
    train_datagen = ImageDataGenerator(#shear_range=0.2,
                                       #zoom_range=0.2,
                                       #rotation_range=15,  # 15 degrees
                                       #horizontal_flip=True,
                                       data_format="channels_last",
                                       rescale=1.0 / 255.0,  # to be sRGB
                                       validation_split=0.9,  # for part of the data
                                       dtype=tf.float32
                                       )

    # Create a flow from the directory
    train_data_gen = train_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                       subset='training',
                                                       shuffle=True,
                                                       seed=123,
                                                       target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE,
                                                       color_mode="rgb",
                                                       class_mode=None
                                                       )
    return custom_generator(train_data_gen), train_data_gen.samples


def create_val_generator():

    # Validation ImageDataGenerator
    valid_datagen = ImageDataGenerator(data_format="channels_last",
                                       rescale=1.0 / 255.0,  # to be sRGB
                                       validation_split=0.7,
                                       dtype=tf.float32,
                                       )

    # Create a flow from the directory for validation data
    val_data_gen = valid_datagen.flow_from_directory(directory=VAL_DIR,
                                                     subset='training',
                                                     shuffle=True,
                                                     seed=123,
                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE,
                                                     color_mode="rgb",
                                                     class_mode=None
                                                     )

    return custom_generator(val_data_gen), val_data_gen.samples


def create_test_generator():
    test_datagen = ImageDataGenerator(
        data_format="channels_last",
        rescale=1.0 / 255.0,  # to be sRGB
        validation_split=0.9,  # for part of the data
        dtype=tf.float32
    )

    test_data_gen = test_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                     subset='validation',
                                                     shuffle=True,
                                                     seed=123,
                                                     target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE,
                                                     color_mode="rgb",
                                                     class_mode=None
                                                     )

    return custom_generator(test_data_gen), test_data_gen.samples


# Customize generator
def custom_generator(gen):
    while True:
        batch = next(gen)
        lab_batch = rgb2lab(batch)  # convert from srgb to lab
        X_batch = lab_batch[:, :, :, 0] / 50. - 1.  # grayscale layer, between [-1, 1]
        Y_batch = lab_batch[:, :, :, 1:] / 128.  # green-red and blue-yellow layers, normalize between [-1, 1]
        yield (X_batch.reshape(X_batch.shape + (1,)), Y_batch)
