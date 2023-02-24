import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def count_data_items(filenames):
    return np.sum([int(x[:-6].split('-')[-1]) for x in filenames])


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [IMSIZE, IMSIZE, 3])
    image = tf.cast(image, tf.float32) / 255.
    return image


feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'complex': tf.io.FixedLenFeature([], tf.int64),
    'frog_eye_leaf_spot': tf.io.FixedLenFeature([], tf.int64),
    'powdery_mildew': tf.io.FixedLenFeature([], tf.int64),
    'rust': tf.io.FixedLenFeature([], tf.int64),
    'scab': tf.io.FixedLenFeature([], tf.int64),
    'healthy': tf.io.FixedLenFeature([], tf.int64)
}


def read_tfrecord(example, labeled=True):
    example = tf.io.parse_single_example(example, feature_map)
    image = decode_image(example['image'])
    if labeled:
        label = [tf.cast(example[x], tf.float32) for x in CLASSES]
    else:
        label = example['image_name']
    return image, label


def create_dataset(filenames, labeled=True, ordered=True, shuffled=False,
                   repeated=False, cached=False, distributed=True):
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto)
    if not ordered:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda x: read_tfrecord(x, labeled=labeled),
        num_parallel_calls=auto)
    if shuffled:
        dataset = dataset.shuffle(2048, seed=SEED)
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    if distributed:
        dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset


def solve_dataset(train_ids, val_ids):
    train_filenames = []
    for j in train_ids:
        for k in range(len(GCS_PATHS)):
            train_filenames += tf.io.gfile.glob(os.path.join(GCS_PATH_AUG_TRAIN[k], f"fold_{folds[i]}", '*.tfrec'))
    np.random.shuffle(train_filenames)

    val_filenames = []
    for j in val_ids:
        val_filenames += tf.io.gfile.glob(os.path.join(GCS_PATH_TEST[k], f"fold_{folds[i]}", '*.tfrec'))

    train_dataset = create_dataset(train_filenames, ordered=False, shuffled=True, repeated=True)
    val_dataset = create_dataset(val_filenames, cached=True)

    return (train_dataset, count_data_items(train_filenames)), (val_dataset, count_data_items(val_filenames))

