# Script to take the stage1_train_labels.csv file and groupby the patientID that might have multiple binding box (x,y,w,h,target)


import csv
from itertools import islice
from configparser import ConfigParser


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

TRAIN_CSV = 'stage_1_train_labels.csv'
TRAIN_GENERATOR = 'train_gen.csv'

########################################
######         MAIN LOOP          ######
########################################

def main():
    result = {}
    def take(n, iterable):
        return list(islice(iterable, n))
    
    with open(DATA_PATH + TRAIN_CSV) as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        next(csvreader, None)
        for row in csvreader:

            if row[0] in result:
                if row[1]:
                    result[row[0]].append(row[1:])
                else:
                    result[row[0]].append(['0', '0', '0', '0', '0'])
            else:
                if row[1]:
                    result[row[0]] = [row[1:]]
                else:
                    result[row[0]] = [['0', '0', '0', '0', '0']]
                    
    train_generator = 'train_gen.csv'
    csv_columns = ['patientId', 'x1', 'y1', 'width1', 'height1', 'confidence1', \
                            'x2', 'y2', 'width2', 'height2', 'confidence2', \
                            'x3', 'y3', 'width3', 'height3', 'confidence3', \
                            'x4', 'y4', 'width4', 'height4', 'confidence4'        
              ]

    with open(DATA_PATH + TRAIN_GENERATOR, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=csv_columns)
        writer.writeheader()
        for k, v in result.items():

            if (len(v) == 1):
                writer.writerow({'patientId':k, 'x1': v[0][0], 'y1': v[0][1], 'width1': v[0][2], 'height1': v[0][3], 'confidence1': v[0][4]
                                              , 'x2': 0, 'y2': 0, 'width2': 0, 'height2': 0, 'confidence2': 0 \
                                              , 'x3': 0, 'y3': 0, 'width3': 0, 'height3': 0, 'confidence3': 0 \
                                              , 'x4': 0, 'y4': 0, 'width4': 0, 'height4': 0, 'confidence4': 0
                                })
            elif (len(v) == 2):
                writer.writerow({'patientId':k, 'x1': v[0][0], 'y1': v[0][1], 'width1': v[0][2], 'height1': v[0][3], 'confidence1': v[0][4]\
                                              , 'x2': v[1][0], 'y2': v[1][1], 'width2': v[1][2], 'height2': v[1][3], 'confidence2': v[1][4]\
                                              , 'x3': 0, 'y3': 0, 'width3': 0, 'height3': 0, 'confidence3': 0 \
                                              , 'x4': 0, 'y4': 0, 'width4': 0, 'height4': 0, 'confidence4': 0
                                })
            elif (len(v) == 3):
                writer.writerow({'patientId':k, 'x1': v[0][0], 'y1': v[0][1], 'width1': v[0][2], 'height1': v[0][3], 'confidence1': v[0][4]\
                                              , 'x2': v[1][0], 'y2': v[1][1], 'width2': v[1][2], 'height2': v[1][3], 'confidence2': v[1][4]\
                                              , 'x3': v[2][0], 'y3': v[2][1], 'width3': v[2][2], 'height3': v[2][3], 'confidence3': v[2][4]\
                                              , 'x4': 0, 'y4': 0, 'width4': 0, 'height4': 0, 'confidence4': 0
                                })    
            else:
                writer.writerow({'patientId':k, 'x1': v[0][0], 'y1': v[0][1], 'width1': v[0][2], 'height1': v[0][3], 'confidence1': v[0][4]\
                                              , 'x2': v[1][0], 'y2': v[1][1], 'width2': v[1][2], 'height2': v[1][3], 'confidence2': v[1][4]\
                                              , 'x3': v[2][0], 'y3': v[2][1], 'width3': v[2][2], 'height3': v[2][3], 'confidence3': v[2][4]\
                                              , 'x4': v[3][0], 'y4': v[3][1], 'width4': v[3][2], 'height4': v[3][3], 'confidence4': v[3][4]
                                })   
     
#    print(take(20, result.items()))
                    
                    
if __name__ == '__main__':
    main()