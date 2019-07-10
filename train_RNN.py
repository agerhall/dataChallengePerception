# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:19:50 2019

@author: sebbe
"""

import keras
from keras.layers import LSTM, TimeDistrubuted, Dense, Dropout
from keras.models import Sequential

def model_arch(nr_of_classes, input_shape):
    model = Sequential()
    model.add(LSTM(units = 100, return_sequnces = True, input_shape = input_shape))
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(TimeDistrubuted(Dense(nr_of_classes)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
   return model


if __name__ == "__main__":
    nr_of_features = 42
    time_steps = 100
    x_train, y_train, x_val, y_val = get_data(time_steps, nr_of_features)

    epochs  = 50
    batch_size = 64
    input_shape = (100, 42)
    model = model_arch(nr_of_classes = 2, input_shape = input_shape)
    model.fit(X, y)