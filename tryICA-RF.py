#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Jun 23 10:06:54 2020

@author: Rakibul Islam Chowdhury (chowdhr)
Summer Student (Plant Imaging)
Canadian Light Source
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#import the two datasets
dataset = pd.read_csv('pea_seed_dataset.csv')

#gather some idea about the reference spectra and FTIR spectra
dataset.describe()
dataset.head()

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


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


#find out the optimum number of ICs
RMSE = []
MAE = []

y_tmp = y.copy()
for constituent in constituents:
    
    y_ = np.array(y_tmp[constituent])
    tmpRMSE = []
    tmpMAE = []
    for i in range(2,20):
        transformer = FastICA(n_components = i, 
                              random_state = 0,
                              max_iter = 2000)
        
        IC = transformer.fit_transform(x.T)
        M = transformer.mixing_
        
        df = np.append(M,y_.reshape(len(y_),1),axis = 1)
        
        df = pd.DataFrame(data = df,
                          index = samples)
        
        #drop the rows that has missing values in any columns
        df.dropna(inplace = True)
        X = df.iloc[:,:-1].values
        Y = df.iloc[:,-1].values
        
        #regressor = SVR(kernel = 'rbf')
        regressor = RandomForestRegressor(n_estimators = 500, 
                                          max_depth = 50,
                                          min_samples_split = 4,
                                          min_samples_leaf = 1,
                                          max_features = 'log2',
                                          random_state = 0)
        #regressor = DecisionTreeRegressor(random_state = 0)
        #regressor = LinearRegression()
        regressor.fit(X, Y)
    
        y_pred = regressor.predict(X)
    
        tmpRMSE.append(sqrt(mean_squared_error(Y, y_pred)))
        tmpMAE.append(mean_absolute_error(Y, y_pred))
    RMSE.append(tmpRMSE)
    MAE.append(tmpMAE)

#plot the result for optimum IC
tmp = [i for i in range(2,20)]
for i in range(len(constituents)):
    plt.plot(tmp, RMSE[i])
    plt.scatter(tmp,RMSE[i])
    #plt.scatter(tmp,MAE[i])
    #plt.plot(tmp,MAE[i])
    plt.xlabel('#IC')
    plt.ylabel('RMSE')
    
from numpy import savetxt
savetxt('RMSE4ICs.csv', RMSE, delimiter=',')

#model with 8 ICs
n_IC = 8
transformer = FastICA(n_components = n_IC, 
                      random_state = 0,
                      max_iter = 2000)
    
#find the IC on the whole dataset
IC = transformer.fit_transform(x.T)
M = transformer.mixing_

#set the target variable
target = constituents[4] #change the index to fit the model to different constituents
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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotNormal #Xavier initializer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.metrics import RootMeanSquaredError

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

    ann.compile(optimizer = 'adam', loss = 'mse', metrics = [RootMeanSquaredError()])
    return ann

#ann = createNN()

#apply grid search cross validation
regressor = KerasRegressor(build_fn = createNN)
paramGrid = {'batch_size':[16,32,64],
             'nb_epoch':[1500,2000,2500],
             'neurons':[150,200,250],
             'dropOutRate':[0.2, 0.3, 0.4]}
search = GridSearchCV(regressor,paramGrid,cv=10)
tmp = search.fit(X_train,y_train)
tmp.best_params_

#train ann with optimized parameters with X_train,y_train and test using X_test
#for n times and average the result
r2 = []
for _ in range(5):
    ann = createNN(neurons = 200,dropOutRate = 0.3)
    
    ann.fit(X_train, y_train, batch_size = 16, epochs = 1500,verbose = 0)
    y_pred = ann.predict(X_test)
    
    r2.append(r2_score(y_test,y_pred))
    
print(np.mean(r2))
#inverse transform the result
y_pred = sc_y.inverse_transform(y_pred)
y_test = sc_y.inverse_transform(y_test)

#plot predicted vs true value
plt.scatter(y_test,y_pred)
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.title('For '+target+ ' with #IC - ' + str(n_IC))
plt.ylim(0,40)
plt.xlim(0,40)

#apply cross validation
regressorCV = KerasRegressor(build_fn = createNN,
                           epochs = 1500,
                           batch_size = 16,
                           verbose = 0)

from sklearn.model_selection import cross_validate
scores = cross_validate(regressorCV,
                        X_train,y_train, 
                        cv = 10,
                        scoring=['r2','neg_mean_squared_error'],
                        return_train_score = True,
                        n_jobs = -1,
                        verbose = 1)

'''
#apply cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor,X,Y, cv = 5, scoring = 'neg_root_mean_squared_error')
'''

'''
#apply random search cross validation
from sklearn.model_selection import RandomizedSearchCV
distributions = dict(n_estimators = [200,300,400],
                     max_depth = [25,50,100],
                     min_samples_split = [2,4,6],
                     min_samples_leaf = [1,2,4],
                     max_features = ['sqrt','log2'])
regressor = RandomForestRegressor(random_state = 0)
clf = RandomizedSearchCV(regressor,distributions,random_state = 0)
search = clf.fit(X,Y)
search.best_params_
search.best_score_
'''


#save the predion in .csv

savetxt('predicted_' + str(target)+'.csv', y_pred, delimiter=',')

#plot the ICs
for i in range(n_IC):
    plt.plot(wavelengths,IC[:,i])


#save the ICs in .csv
for i in range(n_IC):
    fileName = 'IC'+str(i)+'.csv'
    savetxt(fileName, IC[:,i], delimiter=',')

savetxt('wavenumbers.csv',wavelengths,delimiter=',')

'''
#get the weights
coefficients = regressor.coef_
savetxt('regressionCoeff.csv',coefficients,delimiter=',')
'''
