# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:53:04 2022

@author: AMD
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import numpy as np

class EDA():
    def __init__(self):
        pass
    def plot_graph(self,df):
        plt.figure()
        plt.plot(df['Open'])
        plt.plot(df['High'])
        plt.plot(df['Low'])
        plt.legend(['Opening','High','Low'])
        plt.show()
        
    def error_plot(df,low_limit,upper_limit):
        df_error = df[low_limit:upper_limit]
        y_err = df_error['High']- df_error['Low']
        
        plt.figure()
        plt.errorbar(df_error.index,df_error['Open'], yerr=y_err)
        plt.show()
        
class ModelCreation():
    def __init__(self):
        pass
    def simple_lstm_layer(self,X_train):
        model = Sequential ()
        model.add(Input((np.shape(X_train)[1],1))) # input_length, #features
        model.add(LSTM(64,return_sequences=(True)))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear'))
        model.summary()
        
        return model