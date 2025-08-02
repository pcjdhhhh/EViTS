# -*- coding: utf-8 -*-

import pandas
import random
from scipy import stats
from scipy.io import loadmat
from PIL import Image
import math
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
import os
from tool import *

def generate_random_walk():
    
    loaded_data = np.load('datasets/random_walk_data.npy')
    return loaded_data

def get_time_series(file_name):
    
    #file_name = 'Car'
    train_file_path = 'datasets/' + 'time_series/' + file_name + '/' + 'train.mat'
    test_file_path = 'datasets/' + 'time_series/' + file_name + '/' + 'test.mat'
    train = loadmat(train_file_path)
    test = loadmat(test_file_path)
    train = train['train']
    test = test['test']
    train_label = train[:,0]   
    test_label = test[:,0]     
    train_data = train[:,1:]   
    test_data = test[:,1:]     
   
    [num,dim] = train_data.shape
    for i in range(num):
        temp = stats.zscore(train_data[i],ddof=1)
        train_data[i] = temp
    
    [num,dim] = test_data.shape
    for i in range(num):
        temp = stats.zscore(test_data[i],ddof=1)
        test_data[i] = temp
    res = np.vstack((train_data,test_data))
    return res

def get_data(file_name):
    if file_name=='random_walk':
        return generate_random_walk()
    else:
        return get_time_series(file_name)
    



