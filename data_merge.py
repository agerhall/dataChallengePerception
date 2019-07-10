# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:14:19 2019

@author: sebbe
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import random
import os

def multiply_rows(df, n):
    duplicated_df = pd.DataFrame(np.repeat(df.values, n, axis = 0))
    return duplicated_df

def create_data_list(rootdir):
    joint_file_name = "groundTruth.txt"
    sound_file_name = ""
    complete_df = pd.DataFrame()
    data_list = []
    idx = 0
    for dir_ in next(iter(os.walk(rootdir)))[1]:
        # print(dir_)
        for file_name in os.listdir(rootdir+ dir_):
            if file_name.endswith('.csv'):
                sound_file_name = file_name
        if sound_file_name == "":
            raise Exception('Sound file name did not update')
        
        data = pd.read_csv(rootdir+ dir_+ os.sep + joint_file_name, sep = " ", header = None) # read the joint data
        data = data.rename(columns = {0: "frame"})
        cut_depth = 5
        data = data.iloc[:-cut_depth] # arbitrarily cut away some end points where there is no sound
        nr_of_people = data.iloc[:,1].max()
        if not os.path.isfile(rootdir+ dir_+ os.sep + sound_file_name): # check if the sound file exists
            continue
        print(nr_of_people)

        cov_data = pd.read_csv(rootdir+ dir_+ os.sep + sound_file_name, sep = ",") # read the sound data without duplicates
        cov_data = cov_data.rename(columns = {'Unnamed: 0': "frame"})
        
        first_frame = data.iloc[:,0].min() # find the first frame in the ground truth
        print("first frame: {}".format(first_frame))
        last_frame = data.iloc[:,0].max() + 1 # find the last frame in the ground truth
        print("last frame: {}".format(last_frame))
        cov_data = cov_data.iloc[first_frame:last_frame,:].reset_index(drop = True)
        
        #d_cov_data = multiply_rows(cov_data,nr_of_people)
#         if len(d_cov_data)!=len(data): # check that the lengths of the two data sets match
#             raise Exception("lengths of data and d_cov_data do not match. Got length data: {}, length d_cov_data {}".format(len(data),len(d_cov_data)))
        idx_col = {"idx": np.repeat(idx,len(data), axis = 0)}
        idx_df = pd.DataFrame(idx_col)
        complete_data = pd.merge(data, cov_data, how = "outer", on = "frame")
        if len(complete_data)!=len(data): # check that the lengths of the two data sets match
            raise Exception("lengths of data and d_cov_data do not match. Got length data: {}, length complete_data: {}".format(len(data),len(complete_data)))
        #complete_data = pd.concat([complete_data,idx_df], axis = 1, ignore_index = True) # merge the data
        idx +=1
        for i in xrange(nr_of_people):
            complete_df = pd.concat([complete_df, complete_data[data.iloc[:,1]==i+1]], ignore_index = True)  
            data_list += [complete_data[data.iloc[:,1]==i+1]]
        
    return data_list, complete_df