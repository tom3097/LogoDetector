# -*- coding: utf-8 -*-
"""Fully Convolutional ResNet50 with sliding window (stride 32) for Keras.
"""

from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, Conv2D
from layers import Softmax4D
from keras.models import Model


def FcnResNet50(input_shape=(None, None, 3), classes_no=33, fine_tune=False):
    # load ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # add classifier
    x = base_model.get_layer('activation_49').output
    x = Dropout(0.5)(x)
    x = Conv2D(classes_no, (3, 3), strides=(1, 1))(x)
    x = Conv2D(classes_no, (5, 5), activation='linear')(x)
    x = Softmax4D(axis=-1)(x)

    # create model
    model = Model(inputs=base_model.input, outputs=x)

    # prepare for fine-tuning
    if fine_tune is True:
        for layer in base_model.layers:
            layer.trainable = False

    return model
