import tensorflow as tf
import keras as K
import numpy as np
import os

batch_size = 256
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

    """
    Prints a confusion matrix
    User must pass in an array of class names
    """

    def __init__(self, x, y_true, class_names):
        super().__init__()

        self.x = x
        self.y_true = y_true
        self.num_classes = len(class_names)
        self.class_names = class_names

    def print_table(self, table):
        """
        Pretty print the table
        """
        longest_cols = [
            (max([len(str(row[i])) for row in table]) + 3)
            for i in range(len(table[0]))
        ]
        row_format = "".join(["{:^" + str(longest_col) +
                     "}" for longest_col in longest_cols])

        for row in table:
            print(row_format.format(*row))

    def on_epoch_end(self, epoch, logs=None):

        print("Calculating confusion matrix...")
        predicted = self.model.predict(
            self.x, verbose=1, batch_size=batch_size)
        predicted = np.argmax(predicted, axis=1)
        ground = np.argmax(self.y_true, axis=1)
        cm = sklearn.metrics.confusion_matrix(
            ground, predicted, labels=None, sample_weight=None)

        """
        The rest is just to pretty print the confusion matrix
        """
        # Add column and row totals
        cm = np.append(cm, [np.sum(cm, axis=0)], axis=0)
        cm = np.append(cm, np.transpose([np.sum(cm, axis=1)]), axis=1)

        # Pad table to include headers
        table = np.zeros((cm.shape[0]+3,cm.shape[1]+2), dtype=int)
        table[3:,2:] = cm

        # Convert table to string
        table = np.array(table, dtype=str)

        height, width = np.shape(table)
        table[0,width//2] = "Predicted"
        table[height//2,0] = "Actual"
        table[table == "0"] = ""  # Supress 0s from printing

        i = 2
        for classname in self.class_names:
            table[1,i] = classname
            table[i+1,1] = classname
            i += 1

        longest_cols = [
            (max([len(str(row[i])) for row in table]) + 3)
            for i in range(width)
        ]
        for i in range(2,width):
            table[2,i] = "="*(longest_cols[i]-1)

        table[-1,1] = "Total"
        table[1,-1] = "Total"

        self.print_table(table)


def F1_score(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    F1 = numerator / denominator
    return tf.reduce_mean(F1)


def dice_coef_loss(y_true, y_pred, smooth=1.0):

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

    conv = K.layers.Conv2D(filters=128,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)

    #conv = K.layers.BatchNormalization()(conv)

    conv = K.layers.Conv2D(filters=256,
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
#    from tensorflow.image import resize_images

#    resized = resize_images(image, (resize_height, resize_width))

#    return resized

    import tensorflow as tf

    # Resize the images
    resized = tf.image.resize_images(image, (resize_height, resize_width))

    # Perform normalization on each resized image in the batch
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resized)

tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint(
    "../models/baseline_classifier.h5",
    monitor="val_loss", verbose=1,
    save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=5, verbose=1)

class_labels = ["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"]
confusion_callback = ConfusionMatrix(imgs_test, labels_test, class_labels)

callbacks = [tb_callback, model_callback, early_callback, confusion_callback]

model = simple_lenet()
#model = resnet()

model.load_weights("../models/baseline_classifier.h5")

model.compile(loss="categorical_crossentropy",
              optimizer=K.optimizers.Adam(lr=0.00001),
              metrics=["accuracy"])

model.summary()

class_weights = {0: 0.15, 1: 0.15, 2: 0.7}
print("Class weights = {}".format(class_weights))

model.fit(imgs_train, labels_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          #class_weight=class_weights,
          validation_data=(imgs_test, labels_test),
          callbacks=callbacks)
