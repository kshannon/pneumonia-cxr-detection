
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import os

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed


batch_size = 32
image_shape = (1024,1024,1)
epochs = 12


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation="relu",
                 input_shape=image_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss=[focal_loss(alpha=.25, gamma=2)],
              optimizer="Adam",
              metrics=["accuracy"])

data_path = "../../rsna_data_numpy/"

imgs_train = np.load(os.path.join(data_path, "imgs_train.npy"),
						 mmap_mode="r", allow_pickle=False)
imgs_test = np.load(os.path.join(data_path, "imgs_test.npy"),
						 mmap_mode="r", allow_pickle=False)

labels_train = np.load(os.path.join(data_path, "labels_train.npy"),
						 mmap_mode="r", allow_pickle=False)
labels_test = np.load(os.path.join(data_path, "labels_test.npy"),
						 mmap_mode="r", allow_pickle=False)

tb_callback = keras.callbacks.TensorBoard(log_dir="./logs")
model_callback = keras.callbacks.ModelCheckpoint("../models/baseline_classifier.h5", monitor="val_loss", verbose=1, save_best_only=True)
early_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

model.fit(imgs_train, labels_train,
          epochs=epochs,
          verbose=1,
          validation_data=(imgs_test, labels_test),
          callbacks=[tb_callback, model_callback, early_callback])
