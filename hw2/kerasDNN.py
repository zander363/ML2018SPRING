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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

    return (X_train, Y_train, X_test)

if __name__ == '__main__':
    X_all, Y_all, X_test = load_data('feature/train_X', 'feature/train_Y', 'feature/test_X')
    
    model = Sequential()
    model.add(Dense(input_dim=123,units=41,activation='relu'))
    model.add(Dense(units=1,activation='softmax'))
    model.compile(loss='mse',optimizer=SGD(lr=0.5),metrics=['accuracy'])
    model.fit(X_all,Y_all,batch_size=123,epochs=1000)
    result = model.evaluate(X_all,Y_all)
    print('\n---Test Acc:',result[1])
