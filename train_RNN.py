# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:19:50 2019

@author: sebbe
"""
import numpy as np
import pandas as pd
import keras
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, Activation
from keras.models import Sequential
keras

def model_arch(nr_of_classes, input_shape):
    model = Sequential()
    model.add(LSTM(units = 100, return_sequences = True, recurrent_dropout = 0.5 ,input_shape = input_shape))
    model.add(LSTM(units = 100, return_sequences = True, recurrent_dropout = 0.75))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(nr_of_classes)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    return model

def split_sequences(X_data, y_data, time_steps):
    """
    split a multivariate sequence into samples
    
    Parameters
    ----------
    x_data : np.array
        an array of shape [tim_steps, features] with data for each timesteps
    y_data : np.array
        an array of shape [tim_steps, ] with labels for each timestep
    is_distracted : np.array
        an array of shape [tim_steps, ] with distraction labels for each timestep
    time_steps : int
        Desired number of time_steps per sample
        
    Returns
    ----------
    X : np.array
       An array with size [samples, time_steps, features] with data grouped into samples
    y : np.array
        An array with size [samples,] of labels
    """
    
    X, y = list(), list()
    start_idx=0
    for i in range(len(X_data)):
        # find the end of this pattern
        end_idx = int(start_idx + time_steps)

        
        # check if we are beyond the dataset
        if end_idx > len(X_data):
            break
        # gather input and output parts of the pattern
        seq_x = np.asarray(X_data.iloc[start_idx:end_idx, :])
        seq_y = np.asarray(y_data.iloc[start_idx:end_idx])
        
        norm_seq_x = np.zeros(seq_x.shape)
        # normlize each time frame
#        for j in range(seq_x.shape[1]):
#            norm_seq_x[:,j] = normalize_data(seq_x[:,j])

        # set sample label according to the majority label in the sample
        #seq_y = decide_label(y_data[start_idx:end_idx-1])
        

        X.append(seq_x)
        y.append(seq_y)
        # start the next sample after the previous (possible update: enable overlapping samples)
        start_idx = end_idx +1
    X_array = np.empty((len(X),X[0].shape[0],X[0].shape[1]))
    y_array = np.empty((len(X),X[0].shape[0],2))
    for i, (elx, ely) in enumerate(zip(X,y)):
        X_array[i,:,:] = elx
        for j in range(len(ely)):
            y_array[i,j,0] = int(1>np.asarray(ely)[j])
        
        y_array[i,:,1] = 1 - y_array[i,:,0]
        
    return X_array, y_array

def get_data(time_steps, nr_of_features, split_frac, path):
    data = pd.read_csv(path)
    data = data.fillna(-1)
    

    x = data.iloc[:,4:]
    y = data.iloc[:,3]

    seq_x, seq_y = split_sequences(x, y, time_steps)
    split_idx = round(split_frac*len(seq_x))
    perm = np.random.permutation(len(seq_x)) # create permutation vector
    shuff_x = seq_x[perm]
    shuff_y = seq_y[perm]
    
    train_seq_x = shuff_x[:split_idx]
    train_seq_y = shuff_y[:split_idx]
    
    val_seq_x = shuff_x[split_idx:]
    val_seq_y = shuff_y[split_idx:]
    
    val_seq_y = val_seq_y.reshape((val_seq_y.shape[0],val_seq_y.shape[1],2))
    train_seq_y = train_seq_y.reshape((train_seq_y.shape[0],train_seq_y.shape[1],2))
    

    return train_seq_x, train_seq_y, val_seq_x, val_seq_y
    
    

if __name__ == "__main__":
    nr_of_features = 42
    time_steps = 50
    split_frac = 0.9 # 90% training data
    
    data_path = r'C:\Users\sebbe\OneDrive\Dokument\GitHub\dataChallengePerception\sound_data_meancov\complete_data.csv'
    x_train, y_train, x_val, y_val = get_data(time_steps, nr_of_features, split_frac, data_path)

    epochs  = 50
    batch_size = 64
    input_shape = (time_steps, nr_of_features)
    
    callbacks = []
    model = model_arch(nr_of_classes = 2, input_shape = input_shape)
    model.fit(x_train, y_train, epochs=epochs, batch_size = batch_size, 
              validation_data = (x_val, y_val), callbacks = callbacks, verbose = 1)