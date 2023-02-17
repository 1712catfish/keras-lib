import tensorflow as tf
import pandas as pd
# from keras_cv.layers import RandAugment
import os
import numpy as np
from keras_cv_attention_models import *
from keras import *
from keras.layers import *
import keras
import tensorflow_addons as tfa



train_df, test_df = df_train_test_split(df)

STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
VALIDATION_STEPS = len(test_df) // BATCH_SIZE


df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")

IMSIZE = 224
SEED = 123
BATCH_SIZE = 256
NUM_CLASSES = 12
VALIDATION_SPLIT = 0.2

IMAGE_PATH = GCS_PATH + "/train_images"
CSV_PATH = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"

df = pd.read_csv(CSV_PATH)

classes = list(np.unique(df["labels"].values))
num_classes = len(classes)