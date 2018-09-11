import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 32
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


def define_model(dropout=0.5):

	inputs = K.layers.Input(shape=(None,None,1), name="Images")

	params = dict(kernel_size=(3, 3), activation="relu",
				  padding="same",
				  kernel_initializer="he_uniform")

	params1x1 = dict(kernel_size=(1, 1), activation="relu",
				  padding="same",
				  kernel_initializer="he_uniform")

	fmaps = 16
	conv1 = K.layers.Conv2D(name="conv1a", filters=fmaps, **params1x1)(inputs)
	conv1b = K.layers.Conv2D(name="conv1b", filters=2*fmaps, **params)(conv1)
	conv1b = K.layers.Conv2D(name="conv1c", filters=fmaps, **params1x1)(conv1b)
	conv1a = K.layers.Add(name="add1")([conv1, conv1b])
	pool1 = K.layers.MaxPooling2D(name="pool1", pool_size=(2, 2))(conv1a)

	fmaps = 32
	conv2 = K.layers.Conv2D(name="conv2a", filters=fmaps, **params1x1)(pool1)
	conv2b = K.layers.Conv2D(name="conv2b", filters=2*fmaps, **params)(conv2)
	conv2b = K.layers.Conv2D(name="conv2c", filters=fmaps, **params1x1)(conv2b)
	conv2a = K.layers.Add(name="add2")([conv2, conv2b])
	pool2 = K.layers.MaxPooling2D(name="pool2", pool_size=(2, 2))(conv2a)

	fmaps = 64
	conv3 = K.layers.Conv2D(name="conv3a", filters=fmaps, **params1x1)(pool2)
	conv3b = K.layers.Conv2D(name="conv3b", filters=2*fmaps, **params)(conv3)
	conv3b = K.layers.Conv2D(name="conv3c", filters=fmaps, **params1x1)(conv3b)
	conv3a = K.layers.Add(name="add3")([conv3, conv3b])
	pool3 = K.layers.MaxPooling2D(name="pool3", pool_size=(2, 2))(conv3a)

	fmaps = 128
	conv4 = K.layers.Conv2D(name="conv4a", filters=fmaps, **params1x1)(pool3)
	conv4b = K.layers.Conv2D(name="conv4b", filters=2*fmaps, **params)(conv4)
	conv4b = K.layers.Conv2D(name="conv4c", filters=fmaps, **params1x1)(conv4b)
	conv4a = K.layers.Add(name="add4")([conv4, conv4b])

	conv4b = K.layers.Conv2D(name="1x1", filters=1, **params1x1)(conv4a)

	gap1 = K.layers.GlobalAveragePooling2D()(conv4b)

	prediction = K.layers.Activation(activation="sigmoid")(gap1)

	model = K.models.Model(inputs=[inputs], outputs=[prediction])

	model.compile(loss="binary_crossentropy",
				  optimizer="Adam",
				  metrics=["accuracy", F1_score])

	return model


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint("../models/baseline_classifier.h5", monitor="val_loss", verbose=1, save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)


model = define_model()

model.fit(imgs_train, labels_train,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(imgs_test, labels_test),
		  callbacks=[tb_callback, model_callback, early_callback])
