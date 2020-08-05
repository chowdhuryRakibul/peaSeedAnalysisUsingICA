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
dataset = pd.read_csv('pea_seed_dataset_474_observation.csv')
np.random.seed(0)
rdnSeed = 2
constituentIndex = 0 #[starch, fiber, phytic, carotenoid, protein]
nSet = 1 #which set of k-fold from Peng's model you want to try - 1,2,3
n_IC = 6 #number of ICs

#gather some idea about the reference spectra and FTIR spectra
#dataset.describe()
#dataset.head()

#make a list of samples ID
samples = dataset.iloc[:,0]

#copy the spectra to x and % of constituents in y
x = dataset.iloc[:,1:-5]
y = dataset.iloc[:,-5:]
#collect the wavelengths in the spectra
wavelengths = dataset.columns[1:-5]
#list of constituents
constituents = list(dataset.columns[-5:])


from sklearn.decomposition import FastICA
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
from numpy import savetxt


#find out the ICs on the whole dataset
transformer = FastICA(n_components = n_IC, 
                      random_state = rdnSeed,
                      max_iter = 4000)

IC = transformer.fit_transform(x.T)
M = transformer.mixing_

#set the target variable
target = constituents[constituentIndex] #change the index to fit the model to different constituents
y = np.array(y[target]) 

#create the dataframe so that we can drop the missing values both in x and y
df = np.append(M,y.reshape(len(y),1),axis = 1)
columns_ = np.append(wavelengths,[target]).transpose()

df = pd.DataFrame(data = df)

#drop the rows that has missing values in any columns and reset the index
df.dropna(inplace = True)
df.reset_index(drop = True, inplace = True)


#read training and test ID from Peng's model and create similar test and train dataset 
folderNames = ['starch','fiber','pa','carotenoid','protein']
i = folderNames[constituentIndex]


for j in ['_x_train','_x_test','_y_train','_y_test']:
    for k in range(1,4):
        fileName = 'pea-seed-data-kfold-splitted/'+i+'/set0'+str(k)+j+'.csv'
        pengDF = pd.read_csv(fileName,header = None)
        index = pengDF.iloc[:,0]
        new_ICA = df.drop(index)
        if j == '_x_train' or j =='_x_test':
            ICA_new = new_ICA.iloc[:,:-1]
        else:
            ICA_new = new_ICA.iloc[:,-1]
        newFileName = 'pea-seed-data-kfold-splitted_ICA/'+i+'/set0'+str(k)
        if j == '_x_train':
            tmp = '_x_test'
        elif j =='_x_test':
            tmp = '_x_train'
        elif j=='_y_train':
            tmp = '_y_test'
        elif j == '_y_test':
            tmp = '_y_train'
        newFileName = newFileName + tmp +'.csv'
        savetxt(newFileName, ICA_new, delimiter=',')

#read X and y of the newDataset

fileNameBase = 'pea-seed-data-kfold-splitted_ICA/'+i+'/set0'+str(nSet)
XtrainingFileName = fileNameBase + '_x_train.csv'
XtestFileName = fileNameBase + '_x_test.csv'
YtrainingFileName = fileNameBase + '_y_train.csv'
YtestFileName = fileNameBase + '_y_test.csv'

X_train = pd.read_csv(XtrainingFileName,header= None)
X_test = pd.read_csv(XtestFileName,header= None)

y_train = pd.read_csv(YtrainingFileName,header= None)
y_test = pd.read_csv(YtestFileName,header= None)


'''
#shuffle the dataset
from sklearn.utils import shuffle
df = shuffle(df,random_state = 0)
'''

#apply Min-max scaling to y
from sklearn.preprocessing import MinMaxScaler
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
#y_test = y_test.reshape(-1,1)
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
from tensorflow.keras.regularizers import l1,l2, l1_l2

#the model
def createNN(neurons = 200, dropOutRate = 0.3):
    ann = Sequential()
    ann.add(Dense(units=neurons, 
                  activation='relu',
                  input_dim = M.shape[1],
                  #kernel_regularizer = l1_l2(l1 = 0.0001, l2=0.001),
                  #bias_regularizer = l2(0.01),
                  kernel_initializer = GlorotNormal() ))
    ann.add(Dropout(dropOutRate))
    ann.add(Dense(units=int(neurons/2), activation='relu'))#, kernel_regularizer = l1_l2(l1 = 0.0001, l2 =0.001)))
    ann.add(Dropout(dropOutRate))
    ann.add(Dense(units=int(neurons/4), activation='relu'))#,kernel_regularizer = l1_l2(l1 = 0.0001, l2 = 0.001)))
    ann.add(Dropout(dropOutRate))
    
    ann.add(Dense(units=1, activation='linear'))

    ann.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy',RootMeanSquaredError()])
    return ann

#ann = createNN()

#apply grid search cross validation
regressor = KerasRegressor(build_fn = createNN)
paramGrid = {'batch_size':[8,16,32],
             'nb_epoch':[1000,1500,2000],
             'neurons':[100,150,200],
             'dropOutRate':[0.2, 0.3, 0.4]}
search = GridSearchCV(regressor,paramGrid,cv=10)
tmp = search.fit(X_train,y_train)
tmp.best_params_


#run the model for once
ann = createNN(neurons = 200,dropOutRate = 0.2)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_root_mean_squared_error',
                   mode='min',
                   verbose=1,
                   patience=200,
                   restore_best_weights = True,
                   min_delta = 0)


history = ann.fit(X_train, y_train, 
                  batch_size = 8, 
                  epochs = 1500,
                  verbose = 1)
                  #validation_split=0.2,
                  #callbacks=[es])

y_pred = ann.predict(X_test)
    
r2_score(y_test,y_pred)
y_pred = sc_y.inverse_transform(y_pred)
y_test = sc_y.inverse_transform(y_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))

plt.scatter(y_test,y_pred)
plt.title(target)


#plot the training loss
tmp = [i for i in range(len(history.history['loss']))]
plt.plot(tmp,history.history['loss'])
#plt.plot(tmp,history.history['val_loss'])


#save the prediction in .csv
fileName = 'output/ICA-ANN-Pengs_equivalent_dataset/' + folderNames[constituentIndex] +'/'
savetxt(fileName + 'predicted_' + str(target) + '_' + str(nSet) + '.csv', y_pred, delimiter=',')
savetxt(fileName + 'original_' + str(target)  +'_' + str(nSet) + '.csv', y_test, delimiter=',')

json_file = ann.to_json()
fileName = fileName + target + '_' + str(nSet)
with open(fileName +'.json','w') as file:
    file.write(json_file)
ann.save_weights(fileName + '.hdf5')


#plot the ICs
for i in range(n_IC):
    plt.plot(wavelengths,IC[:,i])


#save the ICs in .csv
for i in range(n_IC):
    fileName = 'IC'+str(i)+'.csv'
    savetxt(fileName, IC[:,i], delimiter=',')

savetxt('wavenumbers.csv',wavelengths,delimiter=',')
