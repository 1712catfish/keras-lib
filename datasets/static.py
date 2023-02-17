import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras_cv.layers import RandAugment
from sklearn.preprocessing import MultiLabelBinarizer



def df_train_test_split(df, test_size=0.2, random_state=1712):
    train = df.sample(frac=1 - test_size, random_state=random_state)
    test = df.drop(train.index)
    return train, test


def dataset_from_dataframe(
        df, directory="",
        batch_size=32,
        x_col="image", y_col="labels",
        target_size=(224, 224),
        preprocessing_function=None,
        buffer_size=42, classes=None,
        ignore_order=False, cached=False,
        augmentation=False, split_labels=False,
        **kwargs
):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # train_df, test_df = df_train_test_split(df)
    # classes = list(np.unique(df[y_col].values))

    def parse_function(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, target_size)
        image = tf.cast(image, tf.float32) / 255.
        return image, label

    filenames = [os.path.join(directory, x) for x in df[x_col].values]

    if classes is None:
        labels = df[y_col].values
    else:
        if split_labels:
            labels = [x.split(" ") for x in df[y_col].values]
        else:
            labels = [[x] for x in df[y_col].values]
        labels = MultiLabelBinarizer(classes=classes).fit_transform(labels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if ignore_order:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    if preprocessing_function is not None:
        dataset = dataset.map(lambda x, y: (preprocessing_function(x), y), num_parallel_calls=AUTOTUNE)

    if augmentation:
        dataset = dataset.map(lambda x, y: (RandAugment(value_range=(0, 255))(x), y), num_parallel_calls=AUTOTUNE)

    if cached:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_csv_datasets(csv_path, **kwargs):
    df = pd.read_csv(csv_path)
    train_df, val_df = df_train_test_split(df)
    classes = list(np.unique(df["labels"].values))

    train_dataset = dataset_from_dataframe(
        train_df, split_labels=True, **kwargs
    )

    validation_dataset = dataset_from_dataframe(
        val_df, split_labels=True, cached=True, ignore_order=True, **kwargs
    )

    print(f"Found {len(train_df)} training images belonging to {classes} classes.")
    print(f"Found {len(val_df)} validation images belonging to {classes} classes.")

    steps_per_epoch = len(train_df) // batch_size
    validation_steps = len(val_df) // batch_size

    return train_dataset, validation_dataset, len(train_df), len(val_df)


# def solve_dataset():
#     if CSV_PATH is not None:
#         return create_csv_datasets(
#             CSV_PATH,
#             batch_size=BATCH_SIZE,
#             directory=IMAGE_PATH,
#             target_size=(IMSIZE, IMSIZE),
#             seed=SEED,
#         )
