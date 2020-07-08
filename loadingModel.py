#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:44:09 2020

@author: Rakibul Islam Chowdhury (chowdhr)
Summer Student (Plant Imaging)
Canadian Light Source
"""

from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotNormal #Xavier initializer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.metrics import RootMeanSquaredError


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import the two datasets
dataset = pd.read_csv('pea_seed_dataset.csv')
np.random.seed(0)
rdnSeed = 2
constituent = 4

#gather some idea about the reference spectra and FTIR spectra
#dataset.describe()
#dataset.head()

#make a list of samples
samples = dataset.iloc[:,0]

#copy the spectra to x and constituents in y
x = dataset.iloc[:,1:-5]
y = dataset.iloc[:,-5:]
#collect the wavelengths in the spectra
wavelengths = dataset.columns[1:-5]
#list of constituents
constituents = list(dataset.columns[-5:])

'''
#remove outliers
outliers = [45,235] #given by Kaiyang. There's no 89 in the dataset
df.drop(index=outliers,axis = 0, inplace=True)
'''
'''
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
'''

from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt


n_IC = 8
transformer = FastICA(n_components = n_IC, 
                      random_state = 0,
                      max_iter = 2000)
    
#find the IC on the whole dataset
IC = transformer.fit_transform(x.T)
M = transformer.mixing_

#set the target variable
target = constituents[constituent] #change the index to fit the model to different constituents
y = np.array(y[target]) 

#create the dataframe so that we can drop the missing values both in x and y
df = np.append(M,y.reshape(len(y),1),axis = 1)
columns_ = np.append(wavelengths,[target]).transpose()

df = pd.DataFrame(data = df,
                  index = samples)

#drop the rows that has missing values in any columns
df.dropna(inplace = True)

#shuffle the dataset and split X and Y
from sklearn.utils import shuffle
df = shuffle(df,random_state = 0)
#df.corr()
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

#split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 1/3, 
                                                    random_state = rdnSeed)

#apply Min-max scaling to y
from sklearn.preprocessing import MinMaxScaler
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = y_test.reshape(-1,1)
y_test = sc_y.transform(y_test)

#apply standard scaling to the spectra
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


def createNN(neurons = 200, dropOutRate = 0.3):
    ann = Sequential()
    ann.add(Dense(units=neurons, 
                  activation='relu',
                  input_dim = M.shape[1], 
                  kernel_initializer = GlorotNormal() ))
    ann.add(Dropout(dropOutRate))
    ann.add(Dense(units=int(neurons/2), activation='relu'))
    ann.add(Dropout(dropOutRate))
    ann.add(Dense(units=int(neurons/4), activation='relu'))
    ann.add(Dropout(dropOutRate))
    ann.add(Dense(units=1, activation='linear'))

    ann.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy',RootMeanSquaredError()])
    return ann



model = createNN()

filePath =  '/home/chowdhr/kaiyangProject/peaSeedAnalysis/output/ICA-ANN/' + str(constituent) + '.' + target + '/' +target+'_'+ str(rdnSeed) + '.hdf5'
model.load_weights(filePath)

y_train_pred = model.predict(X_train)

#inverse transform the result
y_train_pred = sc_y.inverse_transform(y_train_pred)
y_train = sc_y.inverse_transform(y_train)


from numpy import savetxt
savetxt('predictedTraining_' + str(target) + '_' + str(rdnSeed) + '.csv', y_train_pred, delimiter=',')
savetxt('originalTraining_' + str(target)  +'_' + str(rdnSeed) + '.csv', y_train, delimiter=',')