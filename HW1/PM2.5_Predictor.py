#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""
this is a Predictor for PM2.5, 
By linear-regression 
"""
import pandas as pd
import numpy as np
import sys 
import matplotlib.pyplot as plt

feature_num = 10
input = str(sys.argv)[0]
output = str(sys.argv)[1]
def train_data(filename):
    data = pd.read_csv(filename)
    PM25 = data.iloc[9::18,3:14]

data = pd.read_csv('train.csv',encoding='ISO-8859-1')
#data = pd.read_csv(input,encoding='ISO-8859-1')
PM25 = data.iloc[9::18,3:14]
feature1 = data.iloc[1::18,3:14]

b = 0 #the constant term in f* 
w1 = np.zeros(9) #the first order term in f*
#w2 = np.array(18) #the second order term in f*
#w3 = np.array(18) #the third order term in f*

#here is the predict function
def f_s(b,w1,x): 
    return b+np.vdot(w1,x)  #+w2*x*x+w3*x*x*x

#here is the loss function
def L(b,w1,x,y):
    sum=(y-(f_s(b,w1,x)))**2
    return sum


def gradient(b,w1,x,y):
    gradient_v = np.zeros(len(w1)+1)
    for i in range(len(y)):
        arg = np.append(x[i],1)
        gradient_v += -2*(y[i]-(b+np.vdot(w,x[i])))*arg
    gradient_v /= len(y)
    return gradient_v


#set all deature is first order and no regulization
eta = 0.5
w = np.zeros(feature_num)
b = 0
x = PM25.iloc[:,0:-1].values.astype(np.float)
y = PM25.iloc[:,-1].values.astype(np.float)
gradient_v = gradient(b,w,x,y)
zeros = np.zeros(feature_num+1)
i = 0
print(gradient_v)
while(i<10 and (gradient_v!=zeros).any()):
    w_p = w
    b_p = b
    eta_t = eta /(i+1)**0.5 
    w = w_p - eta_t*gradient_v[0:-1]
    b = b_p - eta_t*gradient_v[-1]
    gradient_v = gradient(b,w,x,y) 
    print(gradient_v)
    i += 1
