import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from albumentations import ColorJitter, Compose, HorizontalFlip, Affine, VerticalFlip
from typing_extensions import Concatenate


def data_load(data_dir=None, img_height=224, img_width=224, img_path=None, msk_path=None):
    if data_dir:
        images_path = data_dir + '/images/'
        masks_path = data_dir + '/masks/'

    elif img_path and msk_path:
        images_path = img_path
        masks_path = msk_path

    assert data_dir is None or (img_path is None and msk_path is None), "Provide either dataset path or image " \
                                                                        "and mask path separately "

    img_names = os.listdir(images_path)

    masks_names = os.listdir(masks_path)

    X_dataset = np.zeros((len(img_names), img_height, img_width, 3), dtype=np.float32)
    Y_dataset = np.zeros((len(img_names), img_height, img_width), dtype=np.uint8)

    for i in range(len(img_names)):
        img = cv2.imread(images_path + img_names[i])
        img = cv2.resize(img, (img_height, img_width))

        mask = cv2.imread(masks_path + masks_names[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_height, img_width))

        X_dataset[i] = img / 255.0
        Y_dataset[i] = mask

        # Checking datatype of img and mask
        if i == 0:
            print(type(img))
            print(type(mask))
    Y_dataset = np.expand_dims(Y_dataset, axis=-1)

    return X_dataset, Y_dataset


aug_train = Compose([
    HorizontalFlip(),
    VerticalFlip(),
    ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22),
           always_apply=True),
])


def augmentation(img, msk):
    x_train_out = []
    y_train_out = []
    for i in range(len(img)):
        ug = aug_train(image=img[i], mask=msk[i])
        x_train_out.append(ug['image'])
        y_train_out.append(ug['mask'])
    return np.array(x_train_out), np.array(y_train_out)


def input_ready(image, masks, seed_value=0):
    x_train, x_test, y_train, y_test = train_test_split(image, masks,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed_value)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=0.111,
                                                          shuffle=True,
                                                          random_state=seed_value)

    x_train_aug, y_train_aug = augmentation(x_train, y_train)

    x_train_aug = tf.convert_to_tensor(x_train_aug)
    y_train_aug = tf.convert_to_tensor(y_train_aug)

    x_valid = tf.convert_to_tensor(x_valid)
    y_valid = tf.convert_to_tensor(y_valid)

    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    return zip(x_train_aug, y_train_aug), zip(x_valid, y_valid), zip(x_test, y_test)
