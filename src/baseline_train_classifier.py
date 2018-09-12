import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 1024
epochs = 12
resize_height = 256  # Resize images to this height
resize_width = 256   # Resize images to this width

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
   F1 = numerator / denominator
   return tf.reduce_mean(F1)

def dice_coef_loss(y_true, y_pred, smooth=1.):

	y_true_f = K.backend.flatten(y_true)
	y_pred_f = K.backend.flatten(y_pred)
	intersection = K.backend.sum(y_true_f * y_pred_f)
	loss = -K.backend.log(2. * intersection + smooth) + \
		K.backend.log((K.backend.sum(y_true_f) +
					   K.backend.sum(y_pred_f) + smooth))

	return loss



def simple_lenet():

	inputs = K.layers.Input(shape=(None,None,1), name="Images")

	inputR = K.layers.Lambda(resize_normalize,
							 input_shape=(None, None, 1),
							 output_shape=(resize_height, resize_width,
							 1))(inputs)

	conv = K.layers.Conv2D(filters=32,
						   kernel_size=(3, 3),
						   activation="relu",
						   padding="valid",
						   kernel_initializer="he_uniform")(inputR)

	conv = K.layers.Conv2D(filters=64,
						   kernel_size=(3, 3),
						   activation="relu",
						   padding="valid",
						   kernel_initializer="he_uniform")(conv)

	pool = K.layers.MaxPooling2D(pool_size=(2, 2))(conv)
	dropout = K.layers.Dropout(0.25)(pool)

	flat = K.layers.Flatten()(dropout)

	dense1 = K.layers.Dense(256, activation="relu")(flat)
	dropout = K.layers.Dropout(0.25)(dense1)
	prediction = K.layers.Dense(1, activation="sigmoid")(dropout)

	model = K.models.Model(inputs=[inputs], outputs=[prediction])

	return model

def resnet_layer(inputs, fmaps, name):
	"""
	Residual layer block
	"""

	conv1 = K.layers.Conv2D(name=name+"a", filters=fmaps*2,
							kernel_size=(1, 1), activation="linear",
							padding="same",
							kernel_initializer="he_uniform")(inputs)
	conv1b = K.layers.BatchNormalization()(conv1)

	conv1b = K.layers.Conv2D(name=name+"b", filters=fmaps,
							 kernel_size=(3, 3), activation="linear",
							 padding="same",
							 kernel_initializer="he_uniform")(conv1b)
	conv1b = K.layers.BatchNormalization()(conv1b)
	conv1b = K.layers.Activation("relu")(conv1b)

	conv1b = K.layers.Conv2D(name=name+"c", filters=fmaps*2,
							 kernel_size=(1, 1), activation="linear",
							 padding="same",
							 kernel_initializer="he_uniform")(conv1b)

	conv_add = K.layers.Add(name=name+"_add")([conv1, conv1b])
	conv_add = K.layers.BatchNormalization()(conv_add)

	pool = K.layers.MaxPooling2D(name=name+"_pool", pool_size=(2, 2))(conv_add)

	return pool


def resnet():

	inputs = K.layers.Input(shape=(None,None,1), name="Images")

	inputR = K.layers.Lambda(resize_normalize,
							 input_shape=(None, None, 1),
							 output_shape=(resize_height, resize_width,
							 1))(inputs)

	pool1 = resnet_layer(inputR, 16, "conv1")
	pool2 = resnet_layer(pool1, 32, "conv2")
	#pool3 = resnet_layer(pool2, 64, "conv3")
	#pool4 = resnet_layer(pool3, 128, "conv4")

	pool = pool2

	conv = K.layers.Conv2D(name="NiN1", filters=256,
						   kernel_size=(1, 1),
						   activation="relu",
						   padding="same",
						   kernel_initializer="he_uniform")(pool)
	conv = K.layers.Dropout(dropout)(conv)

	conv = K.layers.Conv2D(name="NiN2", filters=128,
						   kernel_size=(1, 1),
						   activation="relu",
						   padding="same",
						   kernel_initializer="he_uniform")(conv)
	conv = K.layers.Dropout(dropout)(conv)

	conv = K.layers.Conv2D(name="1x1", filters=1,
						   kernel_size=(1, 1),
						   activation="linear",
						   padding="same",
						   kernel_initializer="he_uniform")(conv)

	gap1 = K.layers.GlobalAveragePooling2D()(conv)

	prediction = K.layers.Activation(activation="sigmoid")(gap1)

	model = K.models.Model(inputs=[inputs], outputs=[prediction])

	return model

def resize_normalize(image):
	"""
	Resize images on the graph
	"""

	resized = tf.image.resize_images(image, (resize_height, resize_width))

	return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resized)


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint(
				"../models/baseline_classifier.h5",
				monitor="val_loss", verbose=1,
				save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
				patience=3, verbose=1)


model = simple_lenet()
#model = resnet()

model.compile(loss="binary_crossentropy",
			  optimizer=K.optimizers.Adagrad(),
			  metrics=["accuracy", F1_score])

model.fit(imgs_train, labels_train,
		  epochs=epochs,
		  batch_size=batch_size,
		  verbose=1,
		  validation_data=(imgs_test, labels_test),
		  callbacks=[tb_callback, model_callback, early_callback])
