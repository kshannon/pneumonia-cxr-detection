# Script to take the train csv file and create 2 subset csv files: train and validate
# takes as args the % of data to use for the validation set and a random seed, defaults to 10
# The script maintains the data ratio of class 1 to class 0 when constructing the new csvs, regardless of validation size

import sys
import random
import argparse
from configparser import ConfigParser
import numpy as np
import pandas as pd


########################################
######      Argparse Utility      ######
########################################
parser = argparse.ArgumentParser(description='Modify the random seed and % validation dataset size to create')

parser.add_argument('-seed',
                    type=int,
                    action="store",
                    dest="seed",
                    default=10,
                    help='Pass seed to random generator for sampling rows from DF')
parser.add_argument('-validation',
                    type=float,
                    action="store",
                    dest="validation",
                    default=.30,
                    help='pass a % (0.0 - 1.0) for the size of the validation subset to create from the training set.')
args = parser.parse_args()


########################################
######    ConfigParse Utility     ######
########################################
config = ConfigParser()
config.read('../../config/data_path.ini')
try:
    DATA_PATH = config.get('stage_1', 'data_path')

except:
    print('Error reading data_path.ini, try checking data paths in the .ini')
    sys.exit(1)


########################################
######      GLOBAL CONSTANTS      ######
########################################
SEED = args.seed
VALIDATION_SIZE = args.validation
PNEUMONIA = 1
TRAIN_CSV = 'stage_1_train_labels.csv'


########################################
######         MAIN LOOP          ######
########################################
def main():
    # load data csv file into pandas, split into two dataframes for Target = 0/1
    # Randomly draw 25% from both DFs into two other DFs
    # Concat back the old DFs and the two new DFs, calling the first train and new one valid
    # save to csv, giving us both train and validate csvs with a 75/25 split and even class balance between both.
    # from the EDA notebook:
        # Positive: 8964
        # Negative: 20025
        # Ratio of 1.0 to 3.2 pos to neg

    # create DFs
    df = pd.read_csv(DATA_PATH + TRAIN_CSV)  #df.shape = (28989, 6)
    df_1 = df[df['Target'] >= PNEUMONIA].reset_index(drop=True) #shape = (8964, 6)
    df_0 = df[df['Target'] < PNEUMONIA].reset_index(drop=True) #shape = (20025, 6)

    # Create subsamples of both class DFs with an amount = valid_percent
    df_1_valid = df_1.sample(frac=VALIDATION_SIZE,random_state=SEED).sort_index()
    df_0_valid = df_0.sample(frac=VALIDATION_SIZE,random_state=SEED).sort_index()

    # Using the subdsample lets get the symmetric diference or disjoint from the parent sets
    class_1_diff = df_1.index.symmetric_difference(df_1_valid.index).tolist()
    class_0_diff = df_0.index.symmetric_difference(df_0_valid.index).tolist()

    # Create our training DFs based on that subset from above
    df_1_train = df_1.iloc[class_1_diff]
    df_0_train = df_0.iloc[class_0_diff]

    # Check that our subset DFs for train and valid are equal in size to df_1 and df_0
    try:
        assert df_1_valid.shape[0] + df_1_train.shape[0] == df_1.shape[0]
    except:
        print('Subsets class 1 train or validate are not equal to the class 1 data size.')
        sys.exit(1)
    try:
        assert df_0_valid.shape[0] + df_0_train.shape[0] == df_0.shape[0]
    except:
        print('Subsets class 0 train or validate are not equal to the class 0 data size.')
        sys.exit(1)

    #concat DFs
    df_train = pd.concat([df_1_train, df_0_train])
    df_valid = pd.concat([df_1_valid, df_0_valid])

    #check final shapes
    try:
        assert df_train.shape[0] + df_valid.shape[0] == df.shape[0]
    except:
        print('Training or Validation subsets are not equal to your original dataset size!')
        sys.exit(1)

    #Write out DFs to CSVs
    df_train.to_csv(DATA_PATH + 'train.csv',index=False)
    df_valid.to_csv(DATA_PATH + 'validate.csv',index=False)

    for _ in [[df_train,'Training: '],[df_valid,'Validation: ']]:
        label_bool = _[0]['Target'].tolist()
        data_count = len(label_bool)
        positives = np.sum(label_bool)
        print('Size of total data: {}\nHere are the train/valid subsets you created:\n'.format(df.shape[0]))
        print(_[1])
        print('Positive: {}\nNegative: {}'.format(positives,(data_count-positives)))
        print('Ratio of {} to {} pos to neg\n'.format(positives/positives,np.round(data_count/positives,3)))


if __name__ == '__main__':
    main()