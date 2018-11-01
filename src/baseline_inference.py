# baseline inference script

import sys
import os
import csv
from configparser import ConfigParser
import numpy as np
import h5py
from tqdm import tqdm
import tensorflow as tf
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models


########################################
######    ConfigParse Utility     ######
########################################
# config = ConfigParser()
# config.read('../config/data_path.ini')
# try:
#     stage_1 = config.get('stage_1', 'data_path')
# except:
#     print('Error reading data_path.ini, try checking data paths in the .ini')
#     sys.exit(1)

########################################
######      GLOBAL CONSTANTS      ######
########################################
# DATA_PATH = stage_1 #path to stage 1 data from config
DATA_PATH = '../../data/'
HEADER = ['patientId','PredictionString']
PREDICT_PATH = '../predictions/baseline_tony_v2.csv'
# BOX_1 = '200 200 200 600'
# BOX_2 = '600 200 200 600'
CONFIDENCE = '0.9' #? not sure
CHANNELS = 1
IMG_RESIZE_X = 512
IMG_RESIZE_Y = 512
PREDICTIONS = []


#load model
# model = models.load_model('../models/baseline_classifier_good_for_pneumonia.h5')
model = models.load_model('../models/baseline_classifier.h5', compile=False)
# The "compile=False" looks like it won't try to load in the optimizer or other useless stuff at inference time. So should just be able to do a model.predict() after that.


def gen_rand_box():
    def random_num(num):
        return random.randint(0,num)
    x = random_num(IMG_RESIZE_X)
    y = random_num(IMG_RESIZE_Y)
    w = random_num(IMG_RESIZE_X - x)
    h = random_num(IMG_RESIZE_Y - y)
    return x,y,w,h

def prepare_img(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=CHANNELS) # Don't use tf.image.decode_image
    image = tf.image.convert_image_dtype(image, tf.float32) #convert to float values in [0, 1]
    #image = tf.image.resize_images(image, [IMG_RESIZE_X, IMG_RESIZE_Y])
    image = image[np.newaxis,...] #add on that tricky batch axis
    return image

def inference(img_path, model, data_path=None):
    """Send an img and model, preprocess the img to training standards, then return a pred"""
    img = prepare_img(img_path)
    pred_prob = model.predict(img, batch_size=None, steps=1, verbose=0)
    return np.argmax(pred_prob)

def conf():
    return str(round(random.uniform(0.01,0.99),2))

def main():
    directory = os.fsencode(DATA_PATH + 'stage1-test-png/')  #non google cloud = stage_1_test_pngs
    for png in tqdm(os.listdir(directory)):
        filename = os.fsdecode(png)
        patient_id = filename[0:-4] #clip '.png'
        png_path = os.path.join(DATA_PATH + 'stage1-test-png/', filename)
        pred_class = inference(png_path, model)
        if pred_class == 2:
            confidence = conf()
            box = ' '.join(map(str, gen_rand_box()))
            PREDICTIONS.append((patient_id, confidence + ' ' + box))
            # PREDICTIONS.append((patient_id,CONFIDENCE + ' ' + BOX_1 + ' ' + CONFIDENCE + ' ' + BOX_2))
        else: #class 1 or 0
            PREDICTIONS.append((patient_id,None))

    with open(PREDICT_PATH,'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow([HEADER[0],HEADER[1]])
        for result in PREDICTIONS:
            writer.writerow([result[0],result[1]])

if __name__ == '__main__':
    main()
