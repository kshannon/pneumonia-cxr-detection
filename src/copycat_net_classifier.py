# using code from: https://github.com/jrieke/shape-detection/blob/master/color-multiple-shapes.ipynb

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten

# divide by 128 bounding box


######## Helper Functions ########
# Flip bboxes during training.
# Note: The validation loss is always quite big here because we don't flip the bounding boxes for the validation data.
def IOU(bbox1, bbox2, bbox3, bbox4):
    '''
    Calculate overlap between four bounding boxes [x, y, w, h]
    as the area of intersection over the area of unity
    '''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    x3, y3, w3, h3 = bbox3[0], bbox3[1], bbox3[2], bbox3[3]
    x4, y4, w4, h4 = bbox4[0], bbox4[1], bbox4[2], bbox4[3]
    w_I = min(x1 + w1, x2 + w2, x3 + w3, x4 + w4) - max(x1, x2, x3, x4)
    h_I = min(y1 + h1, y2 + h2, y3 + h3, y4 + h4) - max(y1, y2, y3, y4)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I
    U = w1 * h1 + w2 * h2 - I
    return I / U

def dist(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2] - bbox3[:2] - bbox4[:2])))

def instantiate_model():
    #TODO change input sizes to match out image
    # TODO: Make one run with very deep network (~10 layers).
    filter_size = 3
    pool_size = 2
    # TODO: Maybe remove pooling bc it takes away the spatial information.
    model = Sequential([
            Convolution2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            Convolution2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),
            Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
            Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    # #         MaxPooling2D(pool_size=(pool_size, pool_size)),
            Flatten(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(y.shape[-1])
        ])
    return model


######## Build Model ########

copycat = instantiate_model()
copycat.compile('adadelta', 'mse')

######## Train Model ########

# TODO: Calculate ious directly for all samples (using slices of the array pred_y for x, y, w, h).
for epoch in range(num_epochs_flipping):
    print 'Epoch', epoch
    copycat.fit(train_X, flipped_train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
    pred_y = copycat.predict(train_X)

    for sample, (pred, exp) in enumerate(zip(pred_y, flipped_train_y)):

        # TODO: Make this simpler.
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_objects, -1)

        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]

        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)
                dists[i, j] = dist(exp_bbox, pred_bbox)
                mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

        new_order = np.zeros(num_objects, dtype=int)

        for i in range(num_objects):
            # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
            ind_exp_bbox, ind_pred_bbox = np.unravel_index(ious.argmax(), ious.shape)
            ious_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
            dists_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
            mses_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
            ious[ind_exp_bbox] = -1  # set iou of assigned bboxes to -1, so they don't get assigned again
            ious[:, ind_pred_bbox] = -1
            new_order[ind_pred_bbox] = ind_exp_bbox

        flipped_train_y[sample] = exp[new_order].flatten()

        flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
        ious_epoch[sample, epoch] /= num_objects
        dists_epoch[sample, epoch] /= num_objects
        mses_epoch[sample, epoch] /= num_objects

        acc_shapes_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
        acc_colors_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))


    # Calculate metrics on test data.
    pred_test_y = copycat.predict(test_X)
    # TODO: Make this simpler.
    for sample, (pred, exp) in enumerate(zip(pred_test_y, flipped_test_y)):

        # TODO: Make this simpler.
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_objects, -1)

        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]

        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)
                dists[i, j] = dist(exp_bbox, pred_bbox)
                mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))

        new_order = np.zeros(num_objects, dtype=int)

        for i in range(num_objects):
            # Find pred and exp bbox with maximum iou and assign them to each other (i.e. switch the positions of the exp bboxes in y).
            ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
            ious_test_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
            dists_test_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
            mses_test_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
            mses[ind_exp_bbox] = 1000000#-1  # set iou of assigned bboxes to -1, so they don't get assigned again
            mses[:, ind_pred_bbox] = 10000000#-1
            new_order[ind_pred_bbox] = ind_exp_bbox

        flipped_test_y[sample] = exp[new_order].flatten()

        flipped_test[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
        ious_test_epoch[sample, epoch] /= num_objects
        dists_test_epoch[sample, epoch] /= num_objects
        mses_test_epoch[sample, epoch] /= num_objects

        acc_shapes_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
        acc_colors_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))


    print 'Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.)
    print 'Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch]))
    print 'Mean dist: {}'.format(np.mean(dists_epoch[:, epoch]))
    print 'Mean mse: {}'.format(np.mean(mses_epoch[:, epoch]))
    print 'Accuracy shapes: {}'.format(np.mean(acc_shapes_epoch[:, epoch]))
    print 'Accuracy colors: {}'.format(np.mean(acc_colors_epoch[:, epoch]))

    print '--------------- TEST ----------------'
    print 'Flipped {} % of all elements'.format(np.mean(flipped_test[:, epoch]) * 100.)
    print 'Mean IOU: {}'.format(np.mean(ious_test_epoch[:, epoch]))
    print 'Mean dist: {}'.format(np.mean(dists_test_epoch[:, epoch]))
    print 'Mean mse: {}'.format(np.mean(mses_test_epoch[:, epoch]))
    print 'Accuracy shapes: {}'.format(np.mean(acc_shapes_test_epoch[:, epoch]))
    print 'Accuracy colors: {}'.format(np.mean(acc_colors_test_epoch[:, epoch]))


    # num_epochs_flipping = 50
    # num_epochs_no_flipping = 0  # has no significant effect
    #
    # flipped_train_y = np.array(train_y)
    # flipped = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # ious_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # dists_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # mses_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # acc_shapes_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    # acc_colors_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
    #
    # flipped_test_y = np.array(test_y)
    # flipped_test = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    # ious_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    # dists_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    # mses_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    # acc_shapes_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
    # acc_colors_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
