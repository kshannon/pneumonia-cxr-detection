# baseline DenseNet or VGG19 naive bounding box approach:


########################################
######          IMPORTS           ######
########################################
import sys
import os
import time
from random import randint
import argparse
from configparser import ConfigParser
import h5py
import pathlib
from glob import glob
import pydicom
import numpy as np
import tensorflow as tf
from tensorflow import keras
# tf.enable_eager_execution()
# tfe = tf.contrib.eager


########################################
######      Argparse Utility      ######
########################################
parser = argparse.ArgumentParser(description='Modify the Baseline script!')

parser.add_argument('-epochs',
                    type=int,
                    action="store",
                    dest="epochs",
                    default=30,
                    help='Number of epochs to train model, default = 30')
parser.add_argument('-batch_size',
                    type=int,
                    action="store",
                    dest="batch_size",
                    default=4,
                    help='Number of DICOM imgs to send to the network per batch, default = 4')
args = parser.parse_args()


########################################
######    ConfigParse Utility     ######
########################################
config = ConfigParser()
config.read('../config/data_path.ini')
try:
    stage_1 = config.get('stage_1', 'data_path')
except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)


########################################
######      GLOBAL CONSTANTS      ######
########################################
DATA_PATH = stage_1 #path to stage 1 data from config
IMG_DIR = DATA_PATH + 'stage_1_train_pngs/'
PNG_DIR = DATA_PATH + 'stage_1_train_images/' #using pngs currently over dicom (post dicom decode)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PREFETCH_SIZE = 1
IMG_RESIZE_X = 320
IMG_RESIZE_Y = 320
CHANNELS = 3
LEARNING_RATE = 0.0001
# DECAY_FACTOR = 10 #learning rate decayed when valid. loss plateaus after an epoch
ADAM_B1 = 0.9 #adam optimizer default beta_1 value (Kingma & Ba, 2014)
ADAM_B2 = 0.999 #adam optimizer default beta_2 value (Kingma & Ba, 2014)
MAX_ROTAT_DEGREES = 30 #up to 30 degrees img rotation.
MIN_ROTAT_DEGREES = 0

EXTENSION = '.png' # '.dcm'

TB_LOG_DIR = "../logs/"
CHECKPOINT_FILENAME = "../checkpoints/Baseline_{}.hdf5".format(time.strftime("%Y%m%d_%H%M%S"))# Save Keras model to this file
MODEL_FILENAME = "../models/Baseline_model.h5"


########################################
######        INGEST DATA         ######
########################################
train_paths = DATA_PATH + 'train.csv' #DEBUG
valid_paths = DATA_PATH + 'validate.csv'
print("Using Full train/valid dataset. Data path: '{}'".format(DATA_PATH))


def split_data_labels(csv_path, path):
    """ take CSVs with filepaths/labels and extracts them into parallel lists"""
    filenames = []
    labels = []
    with open(csv_path, 'r') as f:
        next(f)
        for line in f:
            new_line = line.strip().split(',')
            #[0]=patientID (same as DICOM name) [5]=Target
            filenames.append(path + new_line[0]+EXTENSION)
            labels.append(int(new_line[5])) #DEBUG float??? was float before
    return filenames,labels

train_imgs, train_labels = split_data_labels(train_paths, DATA_PATH+PNG_DIR)
valid_imgs, valid_labels = split_data_labels(valid_paths, DATA_PATH+PNG_DIR)


########################################
######      HELPER FUNCTIONS      ######
########################################
def decode(filename, label):
    pass

#TODO deal with DOCOM..... PNG for now
def preprocess_img(filename, label):
    """
    Read filepaths and decode into numerical tensors
    Ensure img is the required dim and has been normalized to ImageNet mean/std
    """
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=CHANNELS) # Don't use tf.image.decode_image
    image = tf.image.convert_image_dtype(image, tf.float32) #convert to float values in [0, 1]
    image = tf.image.resize_images(image, [IMG_RESIZE_X, IMG_RESIZE_Y])
    return image, label

def img_augmentation(image, label):
    """ Call this on minibatch at time of training """
    image = tf.image.random_flip_left_right(image) #lateral inversion with P(0.5)
    #TODO dont think we need rotation because IMGs are 2d and always presented as the coronal plane...
    # image = tf.image.rot90(image, k=randint(0, 4)) #not 0-30 degrees, but 90 degree increments... so sue me!
    image = tf.clip_by_value(image, 0.0, 1.0) #ensure [0.0,1.0] img constraint
    return image, label

def build_dataset(data, labels):
    """todo"""
    labels = tf.one_hot(tf.cast(labels, tf.uint8), 1) #cast labels to dim 2 tf obj
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.repeat()
    # dataset = dataset.map(decode)
    dataset = dataset.map(preprocess_img, num_parallel_calls=2)
    dataset = dataset.map(img_augmentation, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    return dataset


########################################
######        TRAIN LOOP          ######
########################################
def main():
    with tf.device('/cpu:0'):
        print("Building Train/Validation Dataset Objects")
        train_dataset = build_dataset(train_imgs, train_labels)
        valid_dataset = build_dataset(valid_imgs, valid_labels)


    print("Downloading DenseNet PreTrained Weights... Might take ~0:30 seconds")
    DenseNet169 = tf.keras.applications.densenet.DenseNet169(include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(IMG_RESIZE_X, IMG_RESIZE_Y, CHANNELS),
            pooling='max',
            classes=2)
    last_layer = DenseNet169.output
    # print(last_layer)
    preds = tf.keras.layers.Dense(1, activation='sigmoid')(last_layer)
    model = tf.keras.Model(DenseNet169.input, preds)

    # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
            beta1=ADAM_B1,
            beta2=ADAM_B2)

    optimizer_keras = tf.keras.optimizers.Adam(lr=LEARNING_RATE,
            beta_1=ADAM_B1,
            beta_2=ADAM_B2,
            decay=0.10)

    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
            monitor="val_loss",
            verbose=1,
            save_best_only=True)

    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    #TODO custom tensorboard log file names..... or write to unqiue dir as a sub dir in the logs file...
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
            # histogram_freq=1, #this screwed us over... caused tensorboard callback to fail.. why??? DEBUG !!!!!!
            # batch_size=BATCH_SIZE, # and take this out... and boom.. histogam frequency works. sob
            write_graph=True,
            write_grads=False,
            write_images=True)

    print("Compiling Model!")
    model.compile(optimizer=optimizer_keras,
            loss='binary_crossentropy',
            metrics=['accuracy'])

    print("Beginning to Train Model")
    model.fit(train_dataset,
            epochs=EPOCHS,
            steps_per_epoch=(len(train_labels)//BATCH_SIZE), #36808 train number
            verbose=1,
            validation_data=valid_dataset,
            validation_steps= (len(valid_labels)//BATCH_SIZE),  #3197 validation number
            callbacks=[checkpointer,tensorboard])  #https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1


    # Save entire model to a HDF5 file
    model.save(MODEL_FILENAME)
    # # Recreate the exact same model, including weights and optimizer.
    # model = keras.models.load_model('my_model.h5')
    sys.exit()


if __name__ == '__main__':
    main()
