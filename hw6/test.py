from keras.models import load_model
import numpy as np
import pandas as pd
import sys
from CFModel import CFModel
from CFModel import DeepModel
import tensorflow as tf

K_FACTORS = 120
TEST_USER = 3000
MODEL_WEIGHTS_FILE = 'ml1m_weights.h5'

test = sys.argv[1]
output = sys.argv[2]
movie = sys.argv[3]
user = sys.argv[4]

ratings = pd.read_csv(test, usecols=['UserID', 'MovieID'])
dataset = pd.read_csv(test)

'''
ratings = pd.read_csv("train.csv", usecols=['UserID', 'MovieID', 'Rating'])
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print(max_userid)
print(max_movieid)
'''
movies = pd.read_csv(movie,sep="::")
users = pd.read_csv(user,sep="::")

max_userid = users['UserID'].max()
max_movieid = movies['movieID'].max()

model = DeepModel(max_userid, max_movieid, K_FACTORS)

model.load_weights(MODEL_WEIGHTS_FILE)

#model = load_model('model')
#predict = model.predict(X)

def predict_rating(userid, movieid):
    return model.rate(userid, movieid)

prediction = dataset.apply(lambda x: predict_rating(x['UserID'], x['MovieID']), axis=1)
#dataset['prediction'] = model.rate(dataset.UserID,dataset.MovieID)


print(len(prediction))

x=1
csvFile = open(output,'w')
csvFile.write('TestDataID,Rating')
for i in range(len(prediction)):
    csvFile.write('\n' + str(x) + ' , ' + str(prediction[i]))
    x = x+1
csvFile.close()
