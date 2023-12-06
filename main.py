import tensorflow as tf
import numpy as np
import os
import gc
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime
from time import time
from Models.Metrics import dice_loss, prediction_metrics, write_results_file
from Models import ducknet

import sys
# sys.path.append('/content/PolypSegmentation/CustomLayers')
# sys.path.append('/content/PolypSegmentation/DataPreparation')
# sys.path.append('/content/PolypSegmentation')
sys.path.append('C:/DATA/UGA Sem 3/ML for CV/polypsegcode/PolypSegmentation/CustomLayers')
sys.path.append('C:/DATA/UGA Sem 3/ML for CV/polypsegcode/PolypSegmentation/DataPreparation')
sys.path.append('C:/DATA/UGA Sem 3/ML for CV/polypsegcode/PolypSegmentation')

from DataPreparation.ImageLoading import data_load, input_ready

from tensorflow.keras.optimizers import RMSprop

# Measuring run time starting here
time_today = datetime.now()
start = time()

data_dir = 'C:/DATA/UGA Sem 3/ML for CV/polypsegcode/cvcclinicdb'
# data_dir = '/content/PolypSegmentation/Kvasir-SEG'
dataset_type = 'cvc'
model_type = 'duck'

# Alternatively if you have paths for images and masks:
image_path = None
mask_path = None

# Constants
img_height = 352
img_width = 352
seed = 0

# Enter data directory to process and output data ready to be input into model
X, y = data_load(data_dir, img_height=img_height, img_width=img_width, img_path=image_path, msk_path=mask_path,
                 ds_type=dataset_type)

train, valid, test = input_ready(X, y, seed_value=seed)

# Unzipping the input values
image_augmented, mask_augmented = zip(*train)
x_valid, y_valid = zip(*valid)
x_test, y_test = zip(*test)


print(f"Length of training set:{len(image_augmented)}")
print(f"Length of validation set:{len(x_valid)}")
print(f"Length of test set:{len(x_test)}")


image_augmented = tf.convert_to_tensor(image_augmented)
mask_augmented = tf.convert_to_tensor(mask_augmented)

x_valid = tf.convert_to_tensor(x_valid)
y_valid = tf.convert_to_tensor(y_valid)

x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

# print(type(image_augmented))
# print(type(x_valid))
# print(type(x_test))

# Model Parameters
learning_rate = 1e-4
filters = 17  # Number of filters, the paper used 17 and 34
epochs = 2  # Authors use 600
min_loss_for_saving = 0.2

# Creating Model
model = ducknet.create_model(img_height=img_height, img_width=img_width, input_channels=3,
                             out_classes=1,
                             starting_filters=filters)
model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=dice_loss)

# Paths for storing model results
progress_path = 'ProgressFull/' + dataset_type + '_progress_csv_' + model_type + '_filters_' + \
                str(filters) + '_' + str(time_today) + '.csv'

complete_progress_path = 'ProgressFull/' + dataset_type + '_progress_' + model_type + '_filters_' + \
                         str(filters) + '_' + str(time_today) + '.txt'

plot_path = 'ProgressFull/' + dataset_type + '_progress_plot_' + model_type + '_filters_' + \
            str(filters) + '_' + str(time_today) + '.png'

model_path = 'ModelSaveTensorFlow/' + dataset_type + '/' + model_type + '_filters_' + \
             str(filters) + '_' + str(time_today)

final_file = 'results_' + model_type + '_' + str(filters) + '_' + dataset_type + '.txt'

# if not os.path.exists(final_file):
#     os.makedirs(final_file)

# Model Training
for epoch in range(epochs):

    print(f'Training, epoch {epoch}')
    print('Learning Rate: ' + str(learning_rate))

    csv_logger = CSVLogger(progress_path, separator=';', append=True)

    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=4, validation_data=(x_valid, y_valid),
              verbose=1, callbacks=[csv_logger])

    prediction_valid = model.predict(x_valid, verbose=0)
    loss_valid = dice_loss(y_valid, prediction_valid)

    loss_valid = loss_valid.numpy()
    print("Loss Validation: " + str(loss_valid))

    prediction_test = model.predict(x_test, verbose=0)
    loss_test = dice_loss(y_test, prediction_test)
    loss_test = loss_test.numpy()
    print("Loss Test: " + str(loss_test))

    with open(complete_progress_path, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\ntest_loss: ' + str(loss_test) + '\n\n\n')

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved model with val_loss: ", loss_valid)
        model.save(model_path)

    del image_augmented
    del mask_augmented

    gc.collect()

print("Loading the model")
model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss': dice_loss()})

dice, iou, precision, recall, accuracy = prediction_metrics(model, train, valid, test)

write_results_file(final_file, dice, iou, precision, recall, accuracy, dataset_type=dataset_type)

end = time.time()
print('Execution time:', (end - start) / 60, 'minutes')
