#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.


import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, Dense, Dropout, Flatten, Convolution2D, Activation, Reshape
from keras.layers.convolutional import ZeroPadding2D #Conv2D,
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta, Adamax
from keras.utils import np_utils

from keras.layers.advanced_activations import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def load_data(path, label=True):
    In = pd.read_csv(path, sep=',', header=0)
    feat = []
    for i in range(len(In['feature'])):
        feat.append(In['feature'][i].split(' '))
        
    X = np.array(feat,dtype = np.floating)

    if label : 
        Y = np.array(In['label'].values)
        return X,Y
    else:
        return X

if __name__ == '__main__':
    w=48
    h=48
    X,Y = load_data('train.csv')
    X = X.reshape(X.shape[0],h,w,1)
    Y = np_utils.to_categorical(Y,7)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(48,48,1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X,Y,batch_size = 100,epochs=8)
    score = model.evaluate(X,Y)
    model.save("model")
    print("\nACC:",score[1])


