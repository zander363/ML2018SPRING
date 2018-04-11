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
from keras.layers import Input, Dense, Dropout, Flatten, Convolution2D, Activation, Reshape
from keras.layers.convolutional import ZeroPadding2D #Conv2D,
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta, Adamax


def load_data(path, label=True):
    In = pd.read_csv(path, sep=',', header=0)
    feat = []
    for i in range(len(In['feature'])):
        feat.append(In['feature'][i].split(' '))
        
    X = np.array(feat,dtype = float)
    X = X/255

    if label : 
        Y = np.array(In['label'].values)
        return X,Y
    else:
        return X

if __name__ == '__main__':
    w=48
    h=48
    X,Y = load_data('train.csv')
    #X = X.reshape(X.shape[0],h,w)
    Y = np_utils.to_categorical(Y,7)

    model = Sequential()

    model.add(Dense(input_dim=h*w,units=512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    for i in range(10):
        model.add(Dense(1024))
        model.add(Activation("relu"))
    model.add(Dense(7))
    model.add(Activation("softmax"))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    model.fit(X,Y,batch_size = 100,epochs=40)
    score = model.evaluate(X,Y)
    model.save("model")
    print("\nACC:",score[1])

    
