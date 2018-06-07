#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.

"""
this is for pca reconstruct
"""

import os
import sys
from os import listdir

import numpy as np
from skimage import io



input_dir = sys.argv[1]
input_names = listdir(input_dir)
print(input_names[:10])
input_file = os.path.join(input_dir,sys.argv[2])

X = []
for name in input_names:
    img = io.imread('{}/{}'.format(input_dir,name))
    img = img.flatten()
    X.append(img)

X_flat = np.reshape(X,(415,-1))
X = np.array(X)
X_mean = np.mean(X,axis=0)

input_img = io.imread(input_file).flatten()
input_mean = np.mean(X_flat,axis=0)
input_center = input_img - input_mean

print("Run the SVD ......")
U, s, V = np.linalg.svd((X-X_mean).transpose(),full_matrices=False)

Eigen = U[:,:4]

'''
for i in range(4):
    E = U[:,i].reshape(600,600,3)
    E -= np.min(E)
    E /= np.max(E)
    E = (E * 255).astype(np.uint8)
    io.imsave('E'+str(i)+'.jpg',E.reshape((600,600,3)))
print((s/np.sum(s))[:4])
'''
intput_center = io.imread(input_file).flatten() - input_mean

weight = np.dot(input_center,Eigen)

re = input_mean + np.dot(weight,Eigen.T)
re -= np.min(re,0)
re /= np.max(re,0)
re = (re*255).astype(np.uint8)

io.imsave('reconstruction.jpg',re.reshape(600,600,3))

