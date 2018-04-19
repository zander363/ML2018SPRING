#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""

"""
import os, sys
import argparse

def load_data(path):
    In = pd.read_csv(path, sep=',', header=0)
    feat = []
    for i in range(len(In['feature'])):
        feat.append(In['feature'][i].split(' '))
        
    X = np.array(feat,dtype = np.float64)
    X = X/255

    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--test_data_path', type=str,
                        default='test.csv', dest='test',
                        help='Path to testing data')
    parser.add_argument('--prediction_path', type=str,
                        default='output', dest='predict',
                        help='Path to save the prediction result')
    '''
    parser.add_argument('--mode', type=str,
                        default='public', dest='mode',
                        help='mode between public and private')
    '''
    opts = parser.parse_args()


    normal = load_normal_param('param')
    X_test = X_test.reshape(X_test.shape[0],48,48,1)
    X_test = load_data(opts.test)
    mu = np.genfromtxt('mu.csv',delimiter=',')
    sigma = np.genfromtxt('sigma.csv',delimiter=',')
    X_test = (X_test - mu) / sigma
    model = load_model('model')

    model.summary()
    output = model.predict_classes(X_test,batch_size=100,verbose=1)

    x = 0
    with open(opts.predict,'w') as csvFile:
    csvFile.write('id,label')
    for i in range(len(output)):
        csvFile.write('\n' + str(x) + ',' + str(output[i]))
        x = x+1

