# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from data import *
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datasetLoader import *
import warnings
from search_function import *
from tool import *
import time
from aeon.distances import dtw_distance,msm_distance,twe_distance,euclidean_distance
warnings.filterwarnings("ignore")


reduced_dim = 8
distance_choice = ['DTW','MSM','TWED']
select_distance = 'MSM'
select_pretrained = 'ConvNeXt'
to_efficiency_file_name = 'efficiency_PIPED_' + select_distance + '.csv'
tune=True
datasets_name = ['Crop','ECG5000','ElectricDevices','StarLightCurves','TwoPatterns','Wafer','random_walk']
for file_name in datasets_name:
    print('----------------file_name: ------------------',file_name)
    to_csv_efficiency_data = {file_name:['BF','EViTS','EucFilter','DSEFilter','GLB','PIPED']}
    to_csv_efficiency_data['search_time'] = [0,0,0,0,0,0]
    data = get_data(file_name)
    if file_name=='random_walk':
        data = data[0:10000,:]
    query_num = math.ceil(data.shape[0] * 0.01) 
    test_data = data[0:query_num,:].copy()
    
    train_data = data[query_num:,:].copy()
    print('train_data: ',train_data.shape)
    print('test_data: ',test_data.shape)

    candidate_num = 0.2 
    candidate_num = math.ceil(train_data.shape[0]*candidate_num)
    print('candidate_num: ',candidate_num)
    
    
   
    k=1
    
   
    s_time = time.time()
    res_BF = brute_force_search(train_data,test_data,k,select_distance)
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][0] = e_time - s_time   
    
    #EviTS的查询时间
    s_time= time.time()
    features_path = 'tune_features/'+ select_pretrained + '/' + select_distance + '/' + file_name + '.npy'
    features = np.load(features_path)
    test_vectors = features[0:query_num,:].copy()
    train_vectors = features[query_num:,:].copy()
    res_EviTS = search_with_filter_and_refine_with_lower_bound(train_data,test_data,k,candidate_num,train_vectors,test_vectors,select_distance)
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][1] = e_time - s_time  
    
    
    
    s_time= time.time()
    res_EucFilter = search_with_filter_and_refine_with_lower_bound(train_data,test_data,k,candidate_num,train_data,test_data,select_distance)
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][2] = e_time - s_time   
    
    
    
    furthest_index = generate_with_random(train_data.shape[0],reduced_dim)
    train_features = vector_representation(train_data,furthest_index,select_distance)  #这一步需要花销挺久的 预处理不算时间
    
    s_time = time.time()
    
   
    
    test_features = np.zeros([test_data.shape[0],reduced_dim])
    if select_distance=='DTW':
        dis = dtw_distance
    elif select_distance=='MSM':
        dis = msm_distance
    else:
        dis = twe_distance
    for num_i in range(test_data.shape[0]):
        test_features[num_i,:] = [dis(test_data[num_i,:], train_data[furthest_index[j_dim],:]) for j_dim in range(reduced_dim)]
    res_DSE = search_with_filter_and_refine_with_lower_bound(train_data,test_data,k,candidate_num,train_features,test_features,select_distance)
    
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][3] = e_time - s_time  
    
    
    
    s_time = time.time()
    res_GLB = brute_force_search_with_lower_bound(train_data,test_data,k,select_distance)
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][4] = e_time - s_time   
    
    
    p_num = 6
    train_pips = get_all_pips_of_train_data(train_data,p_num)
    x_range = np.arange(0,train_data.shape[1])
    x_range_zscore = stats.zscore(x_range,ddof=1)
    train_pips = get_all_pips_of_train_data(train_data,p_num)

    values,index = get_pips_values_and_index(train_data,x_range_zscore,train_pips)
    s_time = time.time()
    res_PIP = search_with_pips_with_lower_bound(train_data,test_data,k,candidate_num,values,index,p_num,x_range_zscore,select_distance)
    e_time = time.time()
    print(e_time-s_time)
    to_csv_efficiency_data['search_time'][5] = e_time - s_time   
    
    df = pd.DataFrame(to_csv_efficiency_data)
    df.T.to_csv(to_efficiency_file_name,mode='a')
