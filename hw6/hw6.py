#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.


import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import keras
#from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf

dataset = pd.read_csv("train.csv")

dataset.TrainDataID = dataset.TrainDataID.astype('category').cat.codes.values
dataset.UserID = dataset.UserID.astype('category').cat.codes.values
dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values

train, test = train_test_split(dataset, test_size=0.2)

print(dataset.head())
#print(train.head())
#print(test.head())
users_number, movies_number = len(dataset.UserID.unique()), len(dataset.MovieID.unique())
n_latent_factors = 3

movie_input = keras.layers.Input(shape=[1],name='Item')
movie_embedding = keras.layers.Embedding(movies_number + 1, n_latent_factors, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(users_number + 1, n_latent_factors,name='User-Embedding')(user_input))

prod = keras.layers.merge([movie_vec, user_vec], mode='dot',name='DotProduct')
model = keras.models.Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error')

model.summary()
predict = model.fit([train.UserID, train.MovieID], train.Rating, epochs=100, verbose=1)
model.save("model")
