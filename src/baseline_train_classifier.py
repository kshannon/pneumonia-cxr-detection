import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 32
image_shape = (1024,1024,1)
epochs = 12

data_path = "../../rsna_data_numpy/"

imgs_train = np.load(os.path.join(data_path, "imgs_train.npy"),
						 mmap_mode="r", allow_pickle=False)
imgs_test = np.load(os.path.join(data_path, "imgs_test.npy"),
						 mmap_mode="r", allow_pickle=False)

labels_train = np.load(os.path.join(data_path, "labels_train.npy"),
						 mmap_mode="r", allow_pickle=False)
labels_test = np.load(os.path.join(data_path, "labels_test.npy"),
						 mmap_mode="r", allow_pickle=False)

def F1_score(y_true, y_pred, smooth=1.0):
   intersection = tf.reduce_sum(y_true * y_pred)
   union = tf.reduce_sum(y_true + y_pred)
   numerator = tf.constant(2.) * intersection + smooth
   denominator = union + smooth
   coef = numerator / denominator
   return tf.reduce_mean(coef)

def define_model(input_shape=(1024,1024,1), dropout=0.5):

	inputs = K.layers.Input(shape=input_shape, name="Images")

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="valid",
				  kernel_initializer="he_uniform")

	conv1 = K.layers.Conv2D(name="conv1a", filters=32, **params)(inputs)
	conv1 = K.layers.Conv2D(name="conv1b", filters=32, **params)(conv1)
	pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1)

	conv2 = K.layers.Conv2D(name="conv2a", filters=64, **params)(pool1)
	conv2 = K.layers.Conv2D(name="conv2b", filters=64, **params)(conv2)
	pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2)

	conv3 = K.layers.Conv2D(name="conv3a", filters=128, **params)(pool2)
	# Trying dropout layers earlier on, as indicated in the paper
	conv3 = K.layers.Dropout(dropout)(conv3)
	conv3 = K.layers.Conv2D(name="conv3b", filters=128, **params)(conv3)

	pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3)

	conv4 = K.layers.Conv2D(name="conv4a", filters=256, **params)(pool3)
	# Trying dropout layers earlier on, as indicated in the paper
	conv4 = K.layers.Dropout(dropout)(conv4)
	conv4 = K.layers.Conv2D(name="conv4b", filters=256, **params)(conv4)

	pool4 = K.layers.MaxPooling2D(name="pool4", pool_size=(2, 2))(conv4)

	conv5 = K.layers.Conv2D(name="conv5a", filters=512, **params)(pool4)
	conv5 = K.layers.Conv2D(name="conv5b", filters=512, **params)(conv5)

	gap1 = K.layers.GlobalAverage2D()(conv5)
	conv6 = K.layers.Conv2D(filters=512, kernel_size=(1,1),
							activation="relu", padding="same")(gap1)
	drop1 = K.layers.Dropout(dropout)(conv6)
	conv7 = K.layers.Conv2D(filters=128, kernel_size=(1,1),
							activation="relu", padding="same")(drop1)
	prediction = K.layers.Conv2D(filters=1, kernel_size=(1,1),
								activation="sigmoid")(conv7)

	model = K.models.Model(inputs=[inputs], outputs=[prediction])

	model.compile(loss="binary_crossentropy",
				  optimizer="Adam",
				  metrics=["accuracy", F1_score])

	return model


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint("../models/baseline_classifier.h5", monitor="val_loss", verbose=1, save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)


model = define_model(image_shape)

model.fit(imgs_train, labels_train,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(imgs_test, labels_test),
		  callbacks=[tb_callback, model_callback, early_callback])
