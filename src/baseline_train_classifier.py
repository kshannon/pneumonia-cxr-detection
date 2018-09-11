import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 256
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

def resnet_layer(inputs, fmaps, name):

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="same",
				  kernel_initializer="he_uniform")

	params1x1 = dict(kernel_size=(1, 1), activation="relu",
				  padding="same",
				  kernel_initializer="he_uniform")

	conv1 = K.layers.Conv2D(name=name+"a", filters=fmaps, **params1x1)(inputs)
	conv1b = K.layers.Conv2D(name=name+"b", filters=2*fmaps, **params)(conv1)
	conv1b = K.layers.Conv2D(name=name+"c", filters=fmaps, **params1x1)(conv1b)
	conv_add = K.layers.Add(name=name+"_add")([conv1, conv1b])
	pool = K.layers.MaxPooling2D(name=name+"_pool", pool_size=(2, 2))(conv_add)

	return pool

def define_model(dropout=0.5):

	inputs = K.layers.Input(shape=(None,None,1), name="Images")

	pool1 = resnet_layer(inputs, 16, "conv1")
	pool2 = resnet_layer(pool1, 32, "conv2")
	pool3 = resnet_layer(pool2, 64, "conv3")
	pool4 = resnet_layer(pool3, 128, "conv4")

	params1x1 = dict(kernel_size=(1, 1), activation="relu",
				  padding="same",
				  kernel_initializer="he_uniform")

	conv = K.layers.Conv2D(name="NiN1", filters=256, **params1x1)(pool4)
	conv = K.layers.Dropout(dropout)(conv)
	conv = K.layers.Conv2D(name="NiN2", filters=128, **params1x1)(conv)
	conv = K.layers.Dropout(dropout)(conv)
	conv = K.layers.Conv2D(name="1x1", filters=1, **params1x1)(conv)

	gap1 = K.layers.GlobalAveragePooling2D()(conv)

	prediction = K.layers.Activation(activation="sigmoid")(gap1)

	model = K.models.Model(inputs=[inputs], outputs=[prediction])

	model.compile(loss=dice_coef_loss, #"binary_crossentropy",
				  optimizer="Adam",
				  metrics=["accuracy", F1_score])

	return model


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint(
				"../models/baseline_classifier.h5",
				monitor="val_loss", verbose=1,
				save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
				patience=3, verbose=1)


model = define_model()

model.fit(imgs_train, labels_train,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(imgs_test, labels_test),
		  callbacks=[tb_callback, model_callback, early_callback])
