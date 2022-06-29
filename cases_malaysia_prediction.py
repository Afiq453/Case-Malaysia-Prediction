
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:49:58 2022

@author: AMD
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime
from modules import ModelCreation
#%% Statics
TRAIN_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
#%% DATA LOADING
df_train = pd.read_csv(TRAIN_PATH, index_col = 'date')
df_test = pd.read_csv(TEST_PATH, index_col = 'date')

#%% DATA INSPECTION

df_train.info()
df_test.info()

df_train.isna().sum() # change to numeric firsst
df_train.duplicated().sum() # No duplicates

df_test.isna().sum() # Got 1 NaNs in Cases_new
df_test.duplicated().sum() # No duplicates


#%% DATA CLEANING

train = df_train.copy()
test = df_test.copy()

# CONVERT FROM OBJECT TO NUMERIC
train['cases_new'] = pd.to_numeric(train['cases_new'],
                                         errors='coerce')

# REMOVE NANS USING interpolate
train['cases_new'].interpolate(method='linear', inplace=True)
test['cases_new'].interpolate(method='linear', inplace=True)

train.isna().sum()
test.isna().sum()



#%%

mms = MinMaxScaler()
df= mms.fit_transform(np.expand_dims(train['cases_new'],axis=-1))
test_df = mms.transform(np.expand_dims(test['cases_new'],axis=-1))

X_train = []
y_train = []

win_size = 30 # shape

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)


con_test = np.concatenate((df,test_df), axis=0)
con_test = con_test[-130:] # winsize+size test

X_test = []

for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])

X_test = np.array(X_test)

#%% 


Model = ModelCreation()
model = Model.simple_lstm_layer(X_train)

model.compile(optimizer='adam',
              loss = 'mse',
              metrics='mape')

# %%% Callbacks
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

# %%% Model Training

X_train = np.expand_dims(X_train, -1)
hist = model.fit(X_train, y_train, epochs=24, batch_size=128,
                 callbacks=[tensorboard_callback, early_stopping_callback])

#%%
predicted = model.predict(np.expand_dims(X_test,axis=-1))

# %%% Save model and plot

model.save(MODEL_PATH)
#%%
hist.history.keys()

plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()
#%%
plt.figure()
plt.plot(test_df,'b',label='actual new price')
plt.plot(predicted,'r', label ='predicted new price')
plt.legend()
plt.show()

plt.figure()
plt.plot(mms.inverse_transform(test_df),'b',label='actual new price')
plt.plot(mms.inverse_transform(predicted),'r', label ='predicted new price')
plt.legend()
plt.show()

#%%
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
predicted = mms.inverse_transform(predicted)
test_df = mms.inverse_transform(test_df)

print("𝑀𝑒𝑎𝑛 𝐴𝑏𝑠𝑜𝑙𝑢𝑡𝑒 𝑃𝑒𝑟𝑐𝑒𝑛𝑡𝑎𝑔𝑒 𝐸𝑟𝑟𝑜𝑟 is", mean_absolute_error(test_df, predicted)/sum(abs(test_df))*100)
