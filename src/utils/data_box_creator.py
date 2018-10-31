import sys
import os
import csv
import tqdm
from configparser import ConfigParser



########################################
######    ConfigParse Utility     ######
# ########################################
# config = ConfigParser()
# config.read('../../config/data_path.ini')
# try:
#     DATA_PATH = config.get('stage_1', 'data_path')
#
# except:
#     print('Error reading data_path.ini, try checking data paths in the .ini')
#     sys.exit(1)


########################################
######      GLOBAL CONSTANTS      ######
########################################
TRAIN_CSV = '/Users/keil/datasets/RSNA-pneumonia/stage_1/stage_1_train_labels.csv'
OUT_CSV = './foo.csv'

try:
    os.remove(OUT_CSV)
except OSError:
    pass

def key_checker(bbox_data_dict, row):
     try:
         var1 = bbox_data_dict[row[0]]
     except KeyError as e:
         pass

def main():

    with open('./foo.csv','w') as outfile:
        with open(TRAIN_CSV,'r') as infile:
            bbox_data = {}
            reader = csv.reader(infile)
            header = next(reader)
            for row in reader:
                if row[-1] == '0': #for cases with no pneumonia write out sll boxes as 0s
                    zero_data = row[0] + ',0.0'*20 + '\n'
                    outfile.write(zero_data)
                elif:
                    key_checker(bbox_data, row)
                else:
                    bbox_data.update( {row[0] : row[1:]} )
            for key, value in mydict.items():
                new_data = ',0.0'*5+'/n'
                #TODO check here for num of boxes
                writer.writerow([key, value])



if __name__ == '__main__':
    main()
