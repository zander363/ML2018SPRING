#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.


import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D, MaxPooling2D, Convolution2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils


def load_data(path, label=True):
    In = pd.read_csv(path, sep=',', header=0)
    feat = []
    for i in range(len(In['feature'])):
        feat.append(In['feature'][i].split(' '))
        
    X = np.array(feat,dtype = np.float64)
    X = X/255

    if label : 
        Y = np.array(In['label'].values)
        return X,Y
    else:
        return X

def build_model(shape):
    model = Sequential()

    model.add(ZeroPadding2D(padding=(1,1),input_shape=shape))
    model.add(Convolution2D(64, 5, 5, activation= "relu"))
    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Convolution2D(64, 5, 5, activation= "relu"))
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(BatchNormalization())

    model.add(ZeroPadding2D(padding=(1,1)))
    model.add(Convolution2D(128, 5, 5, activation= "relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 5, 5, activation= "relu"))
    model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(BatchNormalization())

    
    model.add(Flatten())
    model.add(Dense(4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation="softmax"))
    return model

if __name__ == '__main__':
    w=48
    h=48
    X,Y = load_data('train.csv')
    X = X.reshape(X.shape[0],h,w,1)
    Y = np_utils.to_categorical(Y,7)

    model = build_model((h,w,1))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True))

    model.fit(X,Y,batch_size = 100,epochs=5)
    score = model.evaluate(X,Y)
    model.save("model")
    print("\nACC:",score)

    
