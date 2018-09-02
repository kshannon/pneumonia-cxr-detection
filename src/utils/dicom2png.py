# Script takes the training Dicom file folder, crawls through it and converts each DICOM
# into a PNG and writes it out to a new folder. 
# 
# Takes about ~2 hours to run
# Avg DICOM size ~200kb, Avg PNG size ~550kb

import os
import sys
import numpy as np
from configparser import ConfigParser
from tqdm import tqdm
import png
import pydicom



########################################
######    ConfigParse Utility     ######
########################################
config = ConfigParser()
config.read('../../config/data_path.ini')
try:
    stage_1_data = config.get('stage_1', 'data_path')
except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)


########################################
######      GLOBAL CONSTANTS      ######
########################################
DATASETS = []
DATA_PATH = stage_1_data #path to stage 1 data from config
DICOM_TRAIN_DIR = 'stage_1_train_images/'
DICOM_TEST_DIR = 'stage_1_test_images/'
PNG_TRAIN_DIR = 'stage_1_train_pngs/'
PNG_TEST_DIR = 'stage_1_test_pngs/'
TRAIN_CSV = 'stage_1_train_labels.csv'
TEST_CSV = 'stage_1_sample_submission.csv'


# check for png dirs if not create
# for directory in [PNG_TEST_DIR]:   # IF YOU ALREADY RAN THIS CODE ON TRAIN IMAGES ONLY...
for directory in [PNG_TRAIN_DIR, PNG_TEST_DIR]:
    png_dir = DATA_PATH + directory
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)


########################################
######     INGEST DICOM PATHS     ######
########################################
#for csv, directory in zip([TEST_CSV],[DICOM_TEST_DIR]): # IF YOU ALREADY RAN THIS CODE ON TRAIN IMAGES ONLY...
for csv, directory in zip([TRAIN_CSV, TEST_CSV],[DICOM_TRAIN_DIR, DICOM_TEST_DIR]):
    with open(DATA_PATH + csv, 'r') as csv_file:
        next(csv_file)
        dicom_paths = []
        for line in csv_file:
                split_line = line.strip().split(',')
                #[0]=patientID (same as DICOM name), need to save in thus tup: the path to dicom, plus the dicom name to reuse
                dicom_paths.append((DATA_PATH + directory + split_line[0]+'.dcm',split_line[0]))
    DATASETS.append(dicom_paths)


########################################
######    DICOM CONVERT 2 PNG     ######
########################################
def main():
    #var dicom is a tuple containing a [0]=path+dicom, and [1]=dicom img name to use for naming to png
    # for dataset, directory in [(DATASETS, PNG_TEST_DIR)]: # IF YOU ALREADY RAN THIS CODE ON TRAIN IMAGES ONLY...
    for dataset, directory in zip(DATASETS, [PNG_TRAIN_DIR, PNG_TEST_DIR]):
        for dicom in tqdm(dataset[0]):
            # Read in DICOM to python obj
            ds = pydicom.dcmread(dicom[0])
            
            # Convert to float to avoid overflow or underflow losses.
            img = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            img_scaled = (np.maximum(img,0) / img.max()) * 255.0

            # Convert to uint8 dtype
            img_scaled_dtype = np.uint8(img_scaled)

            # Write out PNG to new directory
            with open(DATA_PATH + directory + dicom[1] + '.png', 'wb') as png_file:
                w = png.Writer(img.shape[1], img.shape[0], greyscale=True)
                w.write(png_file, img_scaled_dtype)
            



if __name__ == '__main__':
    main()