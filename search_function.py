# -*- coding: utf-8 -*-
import numpy as np
from aeon.distances import dtw_distance,msm_distance,twe_distance,euclidean_distance
from lower_bound_function import glb_twed,glb_dtw,glb_msm
import random
def brute_force_search(train_data,test_data,k,select_distance):
    
    if select_distance=='DTW':
        dis = dtw_distance
        lb = glb_dtw
    elif select_distance=='MSM':
        dis = msm_distance
        lb = glb_msm
    else:
        dis = twe_distance
        lb = glb_twed

    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf    
        for j in range(n_train):
            temp_dis = dis(query,train_data[j,:])
            min_ = max(min_k)
            if temp_dis<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = j   #
                min_k[location] = temp_dis
    return res

def brute_force_search_with_lower_bound(train_data,test_data,k,select_distance):
    
    if select_distance=='DTW':
        dis = dtw_distance
        lb = glb_dtw
    elif select_distance=='MSM':
        dis = msm_distance
        lb = glb_msm
    else:
        dis = twe_distance
        lb = glb_twed

    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf   
        for j in range(n_train):
            
            min_ = max(min_k)
            lower_bound = lb(query,train_data[j,:])
            if lower_bound<min_:
                temp = dis(query,train_data[j,:])
                if temp<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = j   
                    min_k[location] = temp
    return res


def search_from_first_K_using_save(train_data,test_data,k,select_distance,first_K,DTW_dis_save,MSM_dis_save,TWED_dis_save):
    
    if select_distance=='DTW':
        dis = DTW_dis_save
    elif select_distance=='MSM':
        dis = MSM_dis_save
    else:
        dis = TWED_dis_save
        
    n_test = test_data.shape[0]
    len_first_K = first_K.shape[1]
    res = np.zeros((n_test,k))   
    
    for i in range(n_test):
        #print(i)
        query = test_data[i,:]
        min_k = np.ones(k) * np.inf    
        for j in range(len_first_K):
            temp_dis = dis[i,int(first_K[i,j])]
            min_ = max(min_k)
            if temp_dis<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = int(first_K[i,j])   
                min_k[location] = temp_dis
    return res



def search_with_filter_and_refine(train_data,test_data,k,candidate_num,train_vectors,test_vectors,select_distance):
    
    if select_distance=='DTW':
        dis = dtw_distance
    elif select_distance=='MSM':
        dis = msm_distance
    else:
        dis = twe_distance
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    temp_res = np.zeros((n_test,candidate_num))
    
    
    for i in range(n_test):
        
        query = test_data[i,:]
        vector_query = test_vectors[i,:]
        
       
        vector_dis = np.array([euclidean_distance(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        
        
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp_dis = dis(query,train_data[candidate_index[j],:])
            min_ = max(min_k)
            if temp_dis<min_:
                location = np.where(min_k==min_)[0][0]   
                res[i,location] = candidate_index[j]   
                min_k[location] = temp_dis
    return res


def search_with_filter_and_refine_with_lower_bound(train_data,test_data,k,candidate_num,train_vectors,test_vectors,select_distance):
    
    if select_distance=='DTW':
        dis = dtw_distance
        lb = glb_dtw
    elif select_distance=='MSM':
        dis = msm_distance
        lb = glb_msm
    else:
        dis = twe_distance
        lb = glb_twed
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    temp_res = np.zeros((n_test,candidate_num))
    for i in range(n_test):
       
        query = test_data[i,:]
        vector_query = test_vectors[i,:]
        
       
        vector_dis = np.array([euclidean_distance(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        
        
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            min_ = max(min_k)
           
            lower_bound = lb(query,train_data[candidate_index[j],:])
            
            if lower_bound<min_:
                
                temp = dis(query,train_data[candidate_index[j],:])
                if temp<min_:
                    location = np.where(min_k==min_)[0][0]   
                    res[i,location] = candidate_index[j]   
                    min_k[location] = temp
           
    return res





def search_with_filter_and_refine_using_save(train_data,test_data,k,candidate_num,train_vectors,test_vectors,select_distance,DTW_dis_save,MSM_dis_save,TWED_dis_save):
    
    if select_distance=='DTW':
        dis = DTW_dis_save
    elif select_distance=='MSM':
        dis = MSM_dis_save
    else:
        dis = TWED_dis_save
        
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    res = np.zeros((n_test,k))   
    temp_res = np.zeros((n_test,candidate_num))
    
    
    for i in range(n_test):
        
        query = test_data[i,:]
        vector_query = test_vectors[i,:]
        
        
        vector_dis = np.array([euclidean_distance(vector_query,train_vectors[j,:]) for j in range(n_train)])
        candidate_index = np.argsort(vector_dis)[0:candidate_num]
        
        
        min_k = np.ones(k) * np.inf    
        for j in range(candidate_num):
            temp_dis = dis[i,candidate_index[j]]
            min_ = max(min_k)
            if temp_dis<min_:
                location = np.where(min_k==min_)[0][0]  
                res[i,location] = candidate_index[j]   
                min_k[location] = temp_dis
    return res
