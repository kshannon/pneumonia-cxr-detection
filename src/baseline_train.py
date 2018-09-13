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
from tqdm import tqdm
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
                    default=1,
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
IMG_RESIZE_X = 128
IMG_RESIZE_Y = 128
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
print('\033[95m' + 'Reading & parsing train/validation data...' + '\033[0m')


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
            labels.append(new_line[1:]) #DEBUG float??? was float before ... [1:] is the bbox x,y,w,h,prob
    return filenames,labels

train_imgs, train_labels = split_data_labels(train_paths, DATA_PATH+PNG_DIR)
valid_imgs, valid_labels = split_data_labels(valid_paths, DATA_PATH+PNG_DIR)
print('\033[94m' + 'Data parsed correctly!' + '\033[0m')

########################################
######      HELPER FUNCTIONS      ######
########################################
def decode(filename, label):
    pass

#TODO deal with DICOM..... PNG for now
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
    # image = tf.image.random_flip_left_right(image) #lateral inversion with P(0.5)
    #TODO dont think we need rotation because IMGs are 2d and always presented as the coronal plane...
    # image = tf.image.rot90(image, k=randint(0, 4)) #not 0-30 degrees, but 90 degree increments... so sue me!
    image = tf.clip_by_value(image, 0.0, 1.0) #ensure [0.0,1.0] img constraint
    return image, label

def transform_labels(csv_labels):
    """ Takes csv labels and casts them into tensorflow objects in separate indexed lists"""
    labels,bboxes = [],[]
    for label in tqdm(csv_labels):
        # bbox = [float(x) for x in label[0:-1]]
        bbox = [tf.cast(x, tf.float32) for x in label[0:-1]]
        bboxes.append(bbox)
        class_label = float(label[-1])
        labels.append(tf.cast(class_label, tf.int8))
 
    return labels, bboxes

def build_dataset(data, labels):
    """todo"""
    # class_labels = tf.one_hot(tf.cast(labels[4], tf.uint8), 1) #cast labels to dim 2 tf obj
    labels = transform_labels(labels)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(data))
    dataset = dataset.repeat()
    # dataset = dataset.map(decode)
    dataset = dataset.map(preprocess_img, num_parallel_calls=2)
    # dataset = dataset.map(img_augmentation, num_parallel_calls=2)
    dataset = dataset.batch(BATCH_SIZE) # (?, x, y) unknown batch size because the last batch will have fewer elements.
    dataset = dataset.prefetch(PREFETCH_SIZE) #single training step consumes n elements
    return dataset

def resize_normalize(image):
	""" Resize images on the graph """
	resized = tf.image.resize_images(image, (IMG_RESIZE_X, IMG_RESIZE_Y))
    #TODO tf.image.per-image)standardezation()
	return resized

########################################
######     Model Definitions      ######
########################################
def lenet(img_x,img_y,channels):
    """ Modern take on the 1998 classic by LeCun! """
    #using lambda def()
    input_img = keras.layers.Input(shape=(None,None,1), name='Images')
    # input_img = keras.layers.Input(shape=(img_x,img_y,channels))

    input_resize = K.layers.Lambda(resize_normalize,
    						 input_shape=(None, None, 1),
    						 output_shape=(resize_height, resize_width,
    						 1))(input_img)

    conv1 = keras.layers.Conv2D(filters=20, kernel_size=(5,5))(input_resize)
    conv1 = keras.layers.Activation('relu')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv1)
    dropout1 = keras.layers.Dropout(0.25)(pool1)

    conv2 = keras.layers.Conv2D(filters=50, kernel_size=(5,5))(dropout1)
    conv2 = keras.layers.Activation('relu')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    dropout2 = keras.layers.Dropout(0.25)(pool2)
    flatten = keras.layers.Flatten()(dropout2)
    dense1 = keras.layers.Dense(128)(flatten)

    classification = keras.layers.Dense(1, activation='sigmoid')(dense1)
    bounding_box = keras.layers.Dense(4)(dense1)
    outputs=[classification, bounding_box]

    model = keras.models.Model(inputs=[input_img], outputs=outputs)
    return model



def dense_net169(img_x,img_y,channels,label_shape,classes=2):
    """ todo """
    print("Downloading DenseNet PreTrained Weights... Might take ~0:30 seconds")
    DenseNet169 = tf.keras.applications.densenet.DenseNet169(include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(img_x, img_y, channels),
            pooling='max',
            classes=classes)
    last_layer = DenseNet169.output
    # print(last_layer)
    preds = tf.keras.layers.Dense(label_shape[-1], activation='sigmoid')(last_layer)
    model = tf.keras.Model(DenseNet169.input, preds)
    return model


#####################################
######        TRAIN LOOP          ######
########################################
def main():
    with tf.device('/cpu:0'):
        print('\033[95m' + "Building dataset objects..." + '\033[0m' + '\n')
        train_dataset = build_dataset(train_imgs, train_labels)
        valid_dataset = build_dataset(valid_imgs, valid_labels)
        print('\n' + '\033[94m' + 'tf.dataset objects successfully built!' + '\033[0m')


    #Instantiate MODEL:
    model = lenet(IMG_RESIZE_X,IMG_RESIZE_Y,CHANNELS) #tf.dataset.shape???

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
    #TODO custom tensorboard log file names..... or write to unique dir as a sub dir in the logs file...
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
