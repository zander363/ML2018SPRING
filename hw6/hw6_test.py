#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""

"""
from keras.models import load_model
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

test = sys.argv[1]
output = sys.argv[2]
movie = sys.argv[3]
user = sys.argv[4]

test = pd.read_csv("test.csv")
X = [test.UserID,test.MovieID]

model = load_model('model')
prediction = model.predict(X)

x=1
csvFile = open("output.csv",'w')
csvFile.write('TestDataID,Rating')
for i in range(len(prediction)):
    csvFile.write('\n' + str(x) + ' , ' + str(prediction[i]))
    x = x+1
csvFile.close()
