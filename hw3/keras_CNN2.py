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
from keras.models import load_model


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

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def build_model(shape):
    model = Sequential()

    model.add(Convolution2D(64,kernel_size=(3, 3), input_shape=shape, activation= "relu",padding='same'))
    model.add(Convolution2D(64,kernel_size=(3, 3), activation= "relu",padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(128,kernel_size=(3, 3), activation= "relu",padding='same'))
    model.add(Convolution2D(128,kernel_size=(3, 3), activation= "relu",padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.5))
    '''
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(0.5))
    '''
    model.add(Dense(7,activation="softmax"))
    return model

if __name__ == '__main__':
    w=48
    h=48
    X,Y = load_data('train.csv')
    X_test = load_data('test.csv',False)
    X, X_test = normalize(X,X_test)
    X = X.reshape(X.shape[0],h,w,1)
    X_test = X_test.reshape(X_test.shape[0],h,w,1)
    Y = np_utils.to_categorical(Y,7)

    model = build_model((h,w,1))

    model.summary()
    #model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.5,decay=1e-6,momentum=0.9,nesterov=True),metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X,Y,batch_size = 100,epochs=50, validation_split = 0.1)
    score = model.evaluate(X,Y)
    model.save("model")
    print("\nACC:",score)

    model = load_model('model')
    model.summary()
    output = model.predict_classes(X,batch_size=100,verbose=1)

    x = 0
    with open('output','w') as csvFile:
        csvFile.write('id,label')
        for i in range(len(output)):
            csvFile.write('\n' + str(x) + ',' + str(output[i]))
            x = x+1

