import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

root_dir = "../../.."

path = os.path.join(root_dir, "rsna_data_numpy/")
os.mkdir( path );

df = pd.read_csv(os.path.join(root_dir, "stage_1_train_labels.csv"))

#filename = "training_images/6a35c51e-1d5b-4a08-b863-071916d08d79.dcm"
#filename = "training_images/00704310-78a8-4b38-8475-49f4573b2dbb.dcm"
filename = os.path.join(root_dir, "training_images/00436515-870c-4b36-a041-de91049b9ab4.dcm")
itkimage = sitk.ReadImage(filename)

array = sitk.GetArrayFromImage(itkimage)
array = np.fliplr(np.rot90(np.swapaxes(array,0,2),-1))

# Do a train/test split (85%)
# Note, this will cause some data leakage since multiple
# lines in the CSV refer to the same DICOM file (i.e. > 1 bbox)
msk = np.random.rand(df.shape[0])
len_train = len(np.where(msk < 0.85)[0])
len_test = len(np.where(msk >= 0.85)[0])

imgs_train = np.zeros([len_train, array.shape[0], array.shape[1],1], dtype="float32")
labels_train = np.zeros(len_train, dtype="float32")
bboxes_train = np.zeros([len_train, 4], dtype="float32")

imgs_test = np.zeros([len_test, array.shape[0], array.shape[1],1], dtype="float32")
labels_test = np.zeros(len_test, dtype="float32")
bboxes_test = np.zeros([len_test, 4], dtype="float32")

print("Processing DICOM files and saving to Numpy format")

idx_train = 0
idx_test = 0
idx = 0

with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():

        filename = os.path.join(root_dir, "training_images/" + row.patientId + ".dcm")

        itkimage = sitk.ReadImage(filename)

        array = sitk.GetArrayFromImage(itkimage)
        array = np.fliplr(np.rot90(np.swapaxes(array,0,2),-1))
        #array = (array - np.mean(array)) / np.std(array)
        #plt.imshow(array[:,:,0], cmap="bone")
        #plt.show()

        if msk[idx] < 0.85:  # Train
            imgs_train[idx_train, :, :, :] = array
            labels_train[idx_train] = row.Target
            bboxes_train[idx_train,0] = row.x
            bboxes_train[idx_train,1] = row.y
            bboxes_train[idx_train,2] = row.width
            bboxes_train[idx_train,3] = row.height
            idx_train += 1
        else:
            imgs_test[idx_test, :, :, :] = array
            labels_test[idx_test] = row.Target
            bboxes_test[idx_test,0] = row.x
            bboxes_test[idx_test,1] = row.y
            bboxes_test[idx_test,2] = row.width
            bboxes_test[idx_test,3] = row.height
            idx_test += 1

        idx += 1

        pbar.update(1)

np.save(os.path.join(path, "imgs_train.npy"), imgs_train)
np.save(os.path.join(path, "labels_train.npy"), labels_train)
np.save(os.path.join(path, "bboxes_train.npy"), bboxes_train)
np.save(os.path.join(path, "imgs_test.npy"), imgs_test)
np.save(os.path.join(path, "labels_test.npy"), labels_test)
np.save(os.path.join(path, "bboxes_test.npy"), bboxes_test)
