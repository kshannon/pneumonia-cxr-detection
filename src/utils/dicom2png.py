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
    stage_1 = config.get('stage_1', 'data_path')
    stage_1_img_dir = config.get('stage_1', 'train_img_dir') #DEBUG have a 'train_imgs/' <-- with a slash

except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)


########################################
######      GLOBAL CONSTANTS      ######
########################################
DATA_PATH = stage_1 #path to stage 1 data from config
IMG_DIR = stage_1_img_dir
CSV_FILE = 'stage_1_train_labels.csv'

PNG_IMG_DIR = DATA_PATH + 'stage_1_train_pngs/'
# check for png dir if not create
if not os.path.exists(PNG_IMG_DIR):
    os.makedirs(PNG_IMG_DIR)


########################################
######     INGEST DICOM PATHS     ######
########################################
dicom_paths = []
with open(DATA_PATH + CSV_FILE, 'r') as in_file:
    next(in_file)
    for line in in_file:
            new_line = line.strip().split(',')
            #[0]=patientID (same as DICOM name) [5]=Target
            dicom_paths.append((DATA_PATH + IMG_DIR + new_line[0]+'.dcm',new_line[0]))


########################################
######    DICOM CONVERT 2 PNG     ######
########################################
def main():
    #var dicom is a tuple containing a [0]file path and dicom file name to read, and [1] just the filename to write back out to png
    for dicom in tqdm(dicom_paths):
        # Read in DICOM to python obj
        ds = pydicom.dcmread(dicom[0])
        
        # Convert to float to avoid overflow or underflow losses.
        img = ds.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        img_scaled = (np.maximum(img,0) / img.max()) * 255.0

        # Convert to uint8 dtype
        img_scaled_dtype = np.uint8(img_scaled)

        # Write out PNG to new directory
        with open(PNG_IMG_DIR + dicom[1] + '.png', 'wb') as png_file:
            w = png.Writer(img.shape[1], img.shape[0], greyscale=True)

            w.write(png_file, img_scaled_dtype)
        



if __name__ == '__main__':
    main()