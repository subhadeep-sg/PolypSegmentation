import tensorflow as tf
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt

from datetime import datetime
from time import time
from Models.Metrics import dice_loss, prediction_metrics
from Models import ducknet
from DataPreparation.ImageLoading import data_load, input_ready
from typing_extensions import Concatenate

from tensorflow.keras.optimizers import RMSprop

# Measuring run time starting here
time_today = datetime.now()
start = time()

data_dir = 'C:/DATA/UGA Sem 3/ML for CV/polypsegcode/Kvasir-SEG'
dataset_type = 'kvasir'
model_type = 'duck'

# Alternatively if you have paths for images and masks:
image_path = None
mask_path = None


# Constants
img_height = 244
img_width = 244
seed = 0

# Enter data directory to process and output data ready to be input into model
X, y = data_load(data_dir, img_height=img_height, img_width=img_width, img_path=image_path, msk_path=mask_path)

print(X.shape)

# train, valid, test = input_ready(X, y, seed_value=seed)
#
# print(train.shape)