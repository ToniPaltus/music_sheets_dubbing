import os
# to work with directories
import pathlib
from pathlib import Path

# to work with images
import cv2
import matplotlib.pyplot as plt
# to sort files in directory
import natsort
import numpy as np
import splitfolders
import tensorflow as tf
from PIL import Image
from tensorflow import keras
# model building
from tensorflow.keras import layers
# for tests
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
# image preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def filter_model():
    # ===== RESEARCH =====
    partitures_data_path = "data/partitures/"
    other_data_path = "data/other/"

    # take names and sort them
    input_folder = "data/"

    partitures_filenames = os.listdir(partitures_data_path)
    image_filenames = natsort.natsorted(partitures_filenames)

    other_filenames = os.listdir(other_data_path)
    other_filenames = natsort.natsorted(other_filenames)

    # ===== AUGMENTATION & PREPROCESSING =====
    # split data
    splitfolders.ratio(
        input_folder,
        "data_splited",
        ratio=(0.8, 0.15, 0.05),
        seed=18,
        group_prefix=None,
    )

    # normalization parameters
    train = ImageDataGenerator(rescale=1 / 255)
    val = ImageDataGenerator(rescale=1 / 255)

    # generated normalized images
    train_data = train.flow_from_directory(
        "data_splited/train",
        target_size=(299, 299),
        class_mode="binary",
        batch_size=3,
        shuffle=True,
    )
    val_data = val.flow_from_directory(
        "data_splited/val",
        target_size=(299, 299),
        class_mode="binary",
        batch_size=3,
        shuffle=True,
    )

    # augmentation parameters
    data_augmentation = keras.Sequential(
        [
            # Horizontal
            layers.experimental.preprocessing.RandomFlip(
                "horizontal", input_shape=(299, 299, 3)
            ),
            # Random rotate
            layers.experimental.preprocessing.RandomRotation(0.05),
            # Contrast
            layers.experimental.preprocessing.RandomContrast(0.23),
            # Zoom
            layers.experimental.preprocessing.RandomZoom(0.2),
        ]
    )

    # ===== MODEL BUILDING & LEARNING =====
    model = Sequential(
        [
            data_augmentation,

            layers.Conv2D(16, (3, 3), activation="selu", input_shape=(299, 299, 3)),
            layers.MaxPool2D(2, 2),
            layers.Conv2D(32, (3, 3), activation="selu"),
            layers.MaxPool2D(2, 2),
            layers.Dropout(0.05),

            layers.Conv2D(64, (3, 3), activation="selu"),
            layers.MaxPool2D(2, 2),
            layers.Dropout(0.1),

            layers.Conv2D(128, (2, 2), activation="selu"),
            layers.MaxPool2D(2, 2),
            layers.Conv2D(256, (2, 2), activation="selu"),
            layers.MaxPool2D(2, 2),
            layers.Dropout(0.2),

            layers.Flatten(),
            layers.Dense(500, activation="selu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # model best fit file
    checkpoint_filepath = "checkpoints/best_model2.h5"

    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(lr=0.00024),
        metrics=["binary_accuracy"],
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_binary_accuracy",
        mode="max",
        save_best_only=True,
    )

    history = model.fit(
        train_data,
        batch_size=500,
        verbose=1,
        epochs=35,
        validation_data=val_data,
        callbacks=[model_checkpoint_callback],
    )


if __name__ == "__filter_model__":
    filter_model()
