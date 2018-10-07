import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 512
epochs = 100
resize_height = 512  # Resize images to this height
resize_width = 512   # Resize images to this width

data_path = "../../rsna_data_numpy/"

imgs_train = np.load(os.path.join(data_path, "imgs_train.npy"),
                     mmap_mode="r", allow_pickle=False)
imgs_test = np.load(os.path.join(data_path, "imgs_test.npy"),
                    mmap_mode="r", allow_pickle=False)

labels_train = np.load(os.path.join(data_path, "labels_train.npy"),
                       mmap_mode="r", allow_pickle=False)
labels_test = np.load(os.path.join(data_path, "labels_test.npy"),
                      mmap_mode="r", allow_pickle=False)

from keras.callbacks import Callback
import sklearn


class ConfusionMatrix(Callback):

    def __init__(self, x, y_true, num_classes):
        super().__init__()

        self.x = x
        self.y_true = y_true
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):

        print("Calculating confusion matrix...")
        predicted = self.model.predict(
            self.x, verbose=1, batch_size=batch_size)
        predicted = np.argmax(predicted, axis=1)
        ground = np.argmax(self.y_true, axis=1)
        cm = sklearn.metrics.confusion_matrix(
            ground, predicted, labels=None, sample_weight=None)
        template = "{0:10}|{1:30}|{2:10}|{3:30}|{4:15}|{5:15}"
        print(template.format("", "", "", "Predicted", "", ""))
        print(template.format("", "", "Normal",
                              "No Lung Opacity / Not Normal", "Lung Opacity", "Total true"))
        print(template.format("", "="*28, "="*9, "="*28, "="*12, "="*12))
        print(template.format("", "Normal",
                              cm[0, 0], cm[0, 1], cm[0, 2], np.sum(cm[0, :])))
        print(template.format("True", "No Lung Opacity / Not Normal",
                              cm[1, 0], cm[1, 1], cm[1, 2], np.sum(cm[1, :])))
        print(template.format("", "Lung Opacity",
                              cm[2, 0], cm[2, 1], cm[2, 2], np.sum(cm[2, :])))
        print(template.format("", "Total predicted", np.sum(
            cm[:, 0]), np.sum(cm[:, 1]), np.sum(cm[:, 2]), ""))


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


def sensitivity(y_true, y_pred, smooth=1.):

    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
    return coef


def specificity(y_true, y_pred, smooth=1.):

    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
    return coef


def simple_lenet(dropout_rate=0.5):

    inputs = K.layers.Input(shape=(None, None, 1), name="Images")

    inputR = K.layers.Lambda(resize_normalize,
                             input_shape=(None, None, 1),
                             output_shape=(resize_height, resize_width,1),
                             arguments={"resize_height":resize_height,
                             "resize_width": resize_width})(inputs)


    conv = K.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(inputR)

    #conv = K.layers.BatchNormalization()(conv)

    conv = K.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(conv)

    #conv = K.layers.BatchNormalization()(conv)

    pool = K.layers.MaxPooling2D(pool_size=(2, 2))(conv)

    conv = K.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)

    #conv = K.layers.BatchNormalization()(conv)

    conv = K.layers.Conv2D(filters=128,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(conv)

    #conv = K.layers.BatchNormalization()(conv)

    pool = K.layers.MaxPooling2D(pool_size=(2, 2))(conv)

    flat = K.layers.Flatten()(pool)

    dense1 = K.layers.Dense(256, activation="relu")(flat)
    dropout = K.layers.Dropout(dropout_rate)(dense1)
    dense2 = K.layers.Dense(128, activation="relu")(dense1)
    dropout = K.layers.Dropout(dropout_rate)(dense2)

    prediction = K.layers.Dense(3, activation="softmax")(dropout)

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


def resnet(dropout=0.5):

    inputs = K.layers.Input(shape=(None, None, 1), name="Images")

    inputR = K.layers.Lambda(resize_normalize,
                             input_shape=(None, None, 1),
                             output_shape=(resize_height, resize_width,1),
                             arguments={"resize_height":resize_height,
                             "resize_width": resize_width})(inputs)

    pool1 = resnet_layer(inputR, 16, "conv1")
    pool2 = resnet_layer(pool1, 32, "conv2")
    pool3 = resnet_layer(pool2, 64, "conv3")
    pool4 = resnet_layer(pool3, 128, "conv4")

    pool = pool2

    conv = K.layers.Conv2D(name="NiN2", filters=64,
                           kernel_size=(1, 1),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)
    conv = K.layers.Dropout(dropout)(conv)

    conv = K.layers.Conv2D(name="1x1", filters=3,
                           kernel_size=(1, 1),
                           activation="linear",
                           padding="same",
                           kernel_initializer="he_uniform")(conv)

    gap1 = K.layers.GlobalAveragePooling2D()(conv)

    prediction = K.layers.Activation(activation="softmax")(gap1)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    return model


def resize_normalize(image, resize_height, resize_width):
    """
    Resize images on the graph
    """
    from tensorflow.image import resize_images

    resized = resize_images(image, (resize_height, resize_width))

    return resized


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint(
    "../models/baseline_classifier.h5",
    monitor="val_loss", verbose=1,
    save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=3, verbose=1)

confusion_callback = ConfusionMatrix(imgs_test, labels_test, 3)

callbacks = [tb_callback, model_callback, early_callback, confusion_callback]

model = simple_lenet()
#model = resnet()

model.compile(loss="categorical_crossentropy",
              optimizer=K.optimizers.Adam(lr=0.000001),
              metrics=["accuracy"])

model.summary()

class_weights = {0: 0.1, 1: 0.1, 2: 0.8}
print("Class weights = {}".format(class_weights))

model.fit(imgs_train, labels_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          #class_weight=class_weights,
          validation_data=(imgs_test, labels_test),
          callbacks=callbacks)
