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

X = np.array(X)
X_mean = np.mean(X,axis=0)

U, s, V = np.linalg.svd((X-X_mean).transpose(),full_matrices=False)

Eigen = U[:,:4]
'''
E0 = U[:,0]
E1 = U[:,1]
E2 = U[:,2]
E3 = U[:,3]
E0.reshape((600,600,3))
io.imsave('E0.jpg',E0)
E1.reshape((600,600,3))
io.imsave('E1.jpg',E1)
E2.reshape((600,600,3))
io.imsave('E2.jpg',E2)
E3.reshape((600,600,3))
io.imsave('E3.jpg',E3)
'''
input_img = skimage.io.imread(input_file).flatten()
input_center = input_file - X_mean

weight = np.dot(input_center,Eigen)

re = X_mean + np,dot(weight,Eigen.T)
re -= np.min(re,0)
re /= np.max(re,0)
re = (re*255).astype(np.uint8)


io.imsave('reconstruction.jpg',re.reshape(600,600,3))
