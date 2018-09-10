
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import os


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

model.compile(loss="binary_crossentropy",
              optimizer="Adam",
              metrics=["accuracy"])

data_path = "rsna_data_numpy/"

imgs_train = np.load(os.path.join(data_path, "imgs_train.npy"),
						 mmap_mode="r", allow_pickle=False)
imgs_test = np.load(os.path.join(data_path, "imgs_test.npy"),
						 mmap_mode="r", allow_pickle=False)

labels_train = np.load(os.path.join(data_path, "labels_train.npy"),
						 mmap_mode="r", allow_pickle=False)
labels_test = np.load(os.path.join(data_path, "labels_test.npy"),
						 mmap_mode="r", allow_pickle=False)

tb_callback = keras.callbacks.TensorBoard(log_dir="./logs")
model_callback = keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True)
early_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)

model.fit(imgs_train, labels_train,
          epochs=epochs,
          verbose=1,
          validation_data=(imgs_test, labels_test),
          callbacks=[tb_callback, model_callback, early_callback])
