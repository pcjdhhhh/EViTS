import numpy as np
import math
import matplotlib.pyplot as plt
from preprocessing import *
import pandas as pd
from data import *
import os
import warnings
from aeon.distances import dtw_distance,msm_distance,twe_distance,euclidean_distance
warnings.filterwarnings("ignore")


def vector_representation(train_data,index,select_distance):
   
    if select_distance=='DTW':
        dis = dtw_distance
    elif select_distance=='MSM':
        dis = msm_distance
    else:
        dis = twe_distance
    [n,original_dim] = train_data.shape
    reduced_dim = len(index)
    res_vector = np.zeros((n,reduced_dim))
    
    for i in range(n):
        for j in range(reduced_dim):
            res_vector[i,j] = dis(train_data[i,:], train_data[index[j],:])
    return res_vector

def generate_with_random(n,reduced_dim):
    
    
    
    numbers = np.arange(0, n)
    np.random.shuffle(numbers)
    random_numbers = numbers[0:reduced_dim]
    return random_numbers

def compute_overlapping(true_res,obtained_res):
   
    [n,k] = true_res.shape
    count = 0
    for i in range(n):
        set1 = set(true_res[i,:])
        set2 = set(obtained_res[i,:])
        temp_overrlapping = len(set1 & set2)
        count = count + temp_overrlapping
    return count*1.0 / (n*k)
   

def get_name():
    datasets_df = pd.read_csv('https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv')
    validation_datasets = datasets_df['Name'].values

    varying_length_datasets = ['PLAID', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
                       'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
                        'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
                        'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ']
    
    
    varying_length_datasets = ['PLAID', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
                       'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
                        'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
                        'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ']

    datasets_with_nans = ['DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'MelbournePedestrian']

    validation_datasets = np.setdiff1d(validation_datasets, varying_length_datasets)

    validation_datasets = np.setdiff1d(validation_datasets, datasets_with_nans)
    return validation_datasets



    