# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from data import *
import numpy as np
import os
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




select_pretrained = 'swim_transformer'
with_pca = False
if with_pca:
    results_save_name = select_pretrained + '_with_pca_accuracy.csv'
else:
    results_save_name = select_pretrained + '_no_pca_accuracy.csv'
file_exists = os.path.isfile(results_save_name)
datasets_name = = ['Wafer','TwoPatterns']
distance_choice = ['DTW','MSM','TWED']
for file_name in datasets_name:
    
    data = get_data(file_name)
    
    print(file_name)
    features_path = 'features_' + select_pretrained + '/' + file_name +'_features.npy'
    features = np.load(features_path)

    query_num = math.ceil(data.shape[0] * 0.01)  
    test_data = data[0:query_num,:].copy()
    train_data = data[query_num:,:].copy()
    test_features = features[0:query_num,:].copy()
    train_features = features[query_num:,:].copy()
    
    
    if with_pca:
        
        pca = PCA(n_components=0.95)  
        train_features = pca.fit_transform(train_features)  
        test_features = pca.transform(test_features)
    
    print('train_data: ',train_data.shape)
    print('test_data: ',test_data.shape)

    candidate_num = 0.2 
    candidate_num = math.ceil(train_data.shape[0]*candidate_num)
    print('candidate_num: ',candidate_num)

    
    save_results = {
        file_name:['k=1','k=5','k=10','k=20','k=50'],
        'DTW':list(),
        'TWED':list(),
        'MSM':list()
    }
    
    
    
    DTW_dis_save_path = 'DTW_dis_save/' + file_name + '.txt'
    MSM_dis_save_path = 'MSM_dis_save/' + file_name + '.txt'
    TWED_dis_save_path = 'TWED_dis_save/' + file_name + '.txt'
    if os.path.exists(DTW_dis_save_path):
        print('dis_save exists!')
        
        
        DTW_dis_save = np.loadtxt(DTW_dis_save_path)
        MSM_dis_save = np.loadtxt(MSM_dis_save_path)
        TWED_dis_save = np.loadtxt(TWED_dis_save_path)
    else:
        print('dis_save no exists')
    
    
    
        DTW_dis_save = np.zeros([test_data.shape[0],train_data.shape[0]])
        MSM_dis_save = np.zeros([test_data.shape[0],train_data.shape[0]])
        TWED_dis_save = np.zeros([test_data.shape[0],train_data.shape[0]])
        for i in range(test_data.shape[0]):
            for j in range(train_data.shape[0]):
                DTW_dis_save[i,j] = dtw_distance(test_data[i,:],train_data[j,:])
                MSM_dis_save[i,j] = msm_distance(test_data[i,:],train_data[j,:])
                TWED_dis_save[i,j] = twe_distance(test_data[i,:],train_data[j,:])
        np.savetxt(DTW_dis_save_path, DTW_dis_save)
        np.savetxt(MSM_dis_save_path, MSM_dis_save)
        np.savetxt(TWED_dis_save_path, TWED_dis_save)
    
    
    for i in range(len(distance_choice)):
    
        select_distance = distance_choice[i]
        
    
    
    
        k=50
        KNN_file_path = select_distance + '_KNN_index_results/' + file_name + '_' + str(50)
        if os.path.exists(KNN_file_path):
            print('50-NN exists')
            #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
            brute_force_res_50 = np.loadtxt(KNN_file_path)
            brute_force_res_50 = brute_force_res_50.reshape((len(brute_force_res_50),-1))
            brute_force_res_50.astype('int')
        else:
            print('50-NN no exist')
            s = time.time()
            brute_force_res_50 = brute_force_search_using_save(train_data,test_data,k,select_distance,DTW_dis_save,MSM_dis_save,TWED_dis_save)
            brute_force_res_50.astype('int')
            e = time.time()
            print('brute-force time: ',e-s)
            np.savetxt(KNN_file_path,brute_force_res_50)
            
        kk=[1,5,10,20,50]
        for k in kk:
            print(k)
            
            KNN_file_path = select_distance + '_KNN_index_results/' + file_name + '_' + str(k)
            if os.path.exists(KNN_file_path):
                print(str(k) + '_exists')
                #brute_force_res = np.array([int(np.loadtxt(KNN_file_path))])
                brute_force_res = np.loadtxt(KNN_file_path)
                brute_force_res = brute_force_res.reshape((len(brute_force_res),-1))
            else:
                print(str(k) + '_no exist')
                s = time.time()
                brute_force_res = search_from_first_K_using_save(train_data,test_data,k,select_distance,brute_force_res_50,DTW_dis_save,MSM_dis_save,TWED_dis_save)
                e = time.time()
                print('brute-force time: ',e-s)
                np.savetxt(KNN_file_path,brute_force_res)
            
            
            res_image = search_with_filter_and_refine_using_save(train_data,test_data,k,candidate_num,train_features,test_features,select_distance,DTW_dis_save,MSM_dis_save,TWED_dis_save)
            save_results[select_distance].append(compute_overlapping(brute_force_res,res_image))
            
    df = pd.DataFrame(save_results)
    df.to_csv(results_save_name, mode='a', index=False)
    





