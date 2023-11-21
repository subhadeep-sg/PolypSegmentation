from keras.backend import flatten, cast, sum
import tensorflow as tf
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
import numpy as np


def dice_loss(ground_truth, predictions, smooth=1e-6):
    ground_truth = cast(ground_truth, tf.float32)
    predictions = cast(predictions, tf.float32)

    ground_truth = flatten(ground_truth)
    predictions = flatten(predictions)

    intersection = sum(predictions * ground_truth)
    union = sum(predictions) + sum(ground_truth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


def prediction_metrics(model, train, valid, test):
    x_train = train[0]
    y_train = train[1]

    x_valid = valid[0]
    y_valid = valid[1]

    x_test = test[0]
    y_test = test[1]

    prediction_train = model.predict(x_train, batch_size=4)
    prediction_valid = model.predict(x_valid, batch_size=4)
    prediction_test = model.predict(x_test, batch_size=4)

    print("Predictions done")

    dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                          np.ndarray.flatten(prediction_train > 0.5))
    dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                         np.ndarray.flatten(prediction_test > 0.5))
    dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                          np.ndarray.flatten(prediction_valid > 0.5))
    dice = zip(dice_train, dice_valid, dice_test)

    print("Dice finished")

    mean_iou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                   np.ndarray.flatten(prediction_train > 0.5))
    mean_iou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                                  np.ndarray.flatten(prediction_test > 0.5))
    mean_iou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                   np.ndarray.flatten(prediction_valid > 0.5))
    iou = zip(mean_iou_train, mean_iou_valid, mean_iou_test)

    print("Mean_IoU finished")

    precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                      np.ndarray.flatten(prediction_train > 0.5))
    precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                                     np.ndarray.flatten(prediction_test > 0.5))
    precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                      np.ndarray.flatten(prediction_valid > 0.5))
    precision = zip(precision_train, precision_valid, precision_test)

    print("Precision finished")

    recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                np.ndarray.flatten(prediction_train > 0.5))
    recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                               np.ndarray.flatten(prediction_test > 0.5))
    recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                np.ndarray.flatten(prediction_valid > 0.5))

    recall = zip(recall_train, recall_valid, recall_test)

    print("Recall finished")

    accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                    np.ndarray.flatten(prediction_train > 0.5))
    accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                                   np.ndarray.flatten(prediction_test > 0.5))
    accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                    np.ndarray.flatten(prediction_valid > 0.5))
    accuracy = zip(accuracy_train, accuracy_valid, accuracy_test)

    print("Accuracy finished")

    return dice, iou, precision, recall, accuracy


def write_results_file(final_file, dice, iou, precision, recall, accuracy, dataset_type='kvasir'):

    dice_train, dice_valid, dice_test = zip(*dice)
    iou_train, iou_valid, iou_test = zip(*iou)
    precision_train, precision_valid, precision_test = zip(*precision)
    recall_train, recall_valid, recall_test = zip(*recall)
    accuracy_train, accuracy_valid, accuracy_test = zip(*accuracy)

    with open(final_file, 'a') as f:
        f.write(dataset_type + '\n')
        f.write('dice_train:' + str(dice_train) + ' dice_valid:' + str(dice_valid) + ' dice_test:' + str(dice_test) + '\n\n')
        f.write('mean_iou_train:' + str(iou_train) + ' mean_iou_valid:' + str(iou_valid) + ' mean_iou_test:' + str(iou_test) + '\n\n')
        f.write('precision_train:' + str(precision_train) + ' precision_valid:' + str(precision_valid) + ' precision_test:' + str(
                precision_test) + '\n\n')
        f.write('recall_train:' + str(recall_train) + ' recall_valid:' + str(recall_valid) + ' recall_test:' + str(recall_test) + '\n\n')
        f.write('accuracy_train:' + str(accuracy_train) + ' accuracy_valid:' + str(accuracy_valid) + ' accuracy_test:' + str(accuracy_test) + '\n\n\n\n')

    print('File is ready')
