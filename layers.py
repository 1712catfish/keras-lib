from keras.layers import *


def vh_reduce(inputs):
    print(inputs.shape)
    _, h, w, d = inputs.shape
    x = DepthwiseConv2D(kernel_size=(1, h), padding="valid", activation="linear")(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(w, 1), padding="valid", activation="linear")(x)
    x = BatchNormalization()(x)
    return x
