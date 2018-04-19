#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import ZeroPadding2D, Convolution2D
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
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

def build_model():

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    input_img = Input(shape=(48, 48, 1))
    '''
    先來看一下keras document 的Conv2D
    keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
        padding='valid', data_format=None, dilation_rate=(1, 1),
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)
    '''
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
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

    model = build_model()
    #model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.5,decay=1e-6,momentum=0.9,nesterov=True),metrics=['accuracy'])
    '''
    '''

    model.fit(X,Y,batch_size = 100,epochs=80, validation_split = 0.1)
    score = model.evaluate(X,Y)
    model.save("model")
    print("\nACC:",score)

    model = load_model('model')
    model.summary()
    output = model.predict_classes(X_test,batch_size=100,verbose=1)

    x = 0
    with open('output','w') as csvFile:
        csvFile.write('id,label')
        for i in range(len(output)):
            csvFile.write('\n' + str(x) + ',' + str(output[i]))
            x = x+1

