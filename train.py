import os
import math
import tensorflow as tf
import datetime
from config import BATCH_SIZE, EPOCHS
from data_generators import create_train_generator, create_val_generator
from cnn_model import cnn_model
from keras.callbacks import ModelCheckpoint
os.environ['TF_MIN_LOG_LEVEL'] = '2'  # minimize some log messages

if __name__ == '__main__':

    print("Tensorflow version:", tf.__version__)

    # Check if GPU available
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Create train and validation data generators
    train_data_gen, num_train_samples = create_train_generator()
    val_data_gen, num_val_samples = create_val_generator()

    # Save weights after each epoch
    filepath = './Saved-weights/cnn-model' + '-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False)

    # Tensorboard callback
    log_dir = "./logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Define, compile and fit model
    model = cnn_model()
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(train_data_gen, epochs=EPOCHS, steps_per_epoch=math.ceil(num_train_samples / BATCH_SIZE),
              validation_data=val_data_gen, validation_steps=math.ceil(num_val_samples / BATCH_SIZE), verbose=1,
              callbacks=[checkpoint, tensorboard_callback])

    # Save model
    model.save('./Saved-models/cnn_model_500ep_30data')

    # Open tensorboard
    #  tensorboard - -logdir = "D:\Image colorization project\logs\cnn\20210403 - 210408" --host=127.0.0.1