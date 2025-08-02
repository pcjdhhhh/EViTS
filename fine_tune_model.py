# -*- coding: utf-8 -*-

from model import SwinFeatureExtractor, ViTFeatureExtractor, ResNetFeatureExtractor,ConvNeXtFeatureExtractor
from itertools import combinations
from sklearn.decomposition import PCA
from data import *
import numpy as np
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datasetLoader import *
import torch.optim as optim
import warnings
from search_function import *
from tool import *
import time
from aeon.distances import dtw_distance,msm_distance,twe_distance,euclidean_distance
warnings.filterwarnings("ignore")

select_pretrained = 'swim'


datasets_name = get_name()  
datasets_name = ['random_walk','mixed_sinx']
#datasets_name = [datasets_name[-1]]
distance_choice = ['DTW','MSM','TWED']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train(model, dataloader, optimizer, device, epochs=10):
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for img1, img2, distance in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            distance = distance.to(device)

            f1 = model(img1)
            f2 = model(img2)

            pred_dist = torch.norm(f1 - f2, dim=1)
            loss = criterion(pred_dist, distance)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img1.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

for file_name in datasets_name:
    

    data = get_data(file_name)
    if data.shape[0]<1000:
        continue
    print(file_name)
    query_num = math.ceil(data.shape[0] * 0.01)  
    test_data = data[0:query_num,:].copy()
    train_data = data[query_num:,:].copy()
    
    tune_num = 100
    
    tune_data = data[query_num:query_num+tune_num,:].copy()
    
    
    DTW_tune_save_path = 'DTW_tune_save/' + file_name + '.txt'
    MSM_tune_save_path = 'MSM_tune_save/' + file_name + '.txt'
    TWED_tune_save_path = 'TWED_tune_save/' + file_name + '.txt'
    
    if os.path.exists(DTW_tune_save_path):
        print('tune_dis_save exists!')
        
        
        DTW_tune_save = np.loadtxt(DTW_tune_save_path)
        MSM_tune_save = np.loadtxt(MSM_tune_save_path)
        TWED_tune_save = np.loadtxt(TWED_tune_save_path)
    else:
        print('tune_dis_save no exists')
    
    
    
        DTW_tune_save = np.zeros([tune_num,tune_num])
        MSM_tune_save = np.zeros([tune_num,tune_num])
        TWED_tune_save = np.zeros([tune_num,tune_num])
        for i in range(tune_num):
            for j in range(tune_num):
                DTW_tune_save[i,j] = dtw_distance(tune_data[i,:],tune_data[j,:])
                MSM_tune_save[i,j] = msm_distance(tune_data[i,:],tune_data[j,:])
                TWED_tune_save[i,j] = twe_distance(tune_data[i,:],tune_data[j,:])
        np.savetxt(DTW_tune_save_path, DTW_tune_save)
        np.savetxt(MSM_tune_save_path, MSM_tune_save)
        np.savetxt(TWED_tune_save_path, TWED_tune_save)
    
    
    image_pairs = []
    DTW_pairs_dis = []
    MSM_pairs_dis = []
    TWED_pairs_dis = []
    images_path = 'output_images/' + file_name
    images_name = os.listdir(images_path)
    images_name = images_name[query_num:query_num+tune_num]   
    
    for i,img1 in enumerate(images_name):
        for j,img2 in enumerate(images_name):
            image_pairs.append((img1,img2))
            DTW_pairs_dis.append(DTW_tune_save[i,j])
            MSM_pairs_dis.append(MSM_tune_save[i,j])
            TWED_pairs_dis.append(TWED_tune_save[i,j])
        
    for i in range(3):
        
        select_distance = distance_choice[i]
        dataset = DistanceSupervisedDataset(images_path,image_pairs,DTW_pairs_dis,MSM_pairs_dis,TWED_pairs_dis,select_distance,transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        
        
        if select_pretrained=='swim':
            model = SwinFeatureExtractor().to(device)
        elif select_pretrained=='resnet':
            model = ResNetFeatureExtractor().to(device)
        elif select_pretrained=='ConvNeXt':
            model = ConvNeXtFeatureExtractor().to(device)
        else:
            pass
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        train(model, dataloader, optimizer, device, epochs=10)
        
        model_save_path = 'tune_model_save/' + select_pretrained + '/' + select_distance + '/' + file_name + '.pth'
        torch.save(model.state_dict(), model_save_path)




        

        


