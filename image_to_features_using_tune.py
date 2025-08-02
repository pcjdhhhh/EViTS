# -*- coding: utf-8 -*-

import timm
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
from tool import *
from model import *
warnings.filterwarnings("ignore")

root_dir = 'output_images'  
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distance_choice = ['DTW','MSM','TWED']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

select_pretrained = 'resnet'
datasets_name = get_name()
datasets_name = ['random_walk','mixed_sinx']



for file_name in datasets_name:
    dataset_path = os.path.join(root_dir, file_name)
    
    data_ = get_data(file_name)
    if data_.shape[0]<1000:
        continue
    

    print(f"Processing dataset: {file_name}")

    dataset = ImageFolderDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    for i in range(len(distance_choice)):
    
        select_distance = distance_choice[i]
        
        model_save_path = 'tune_model_save/' + select_pretrained + '/' + select_distance + '/' + file_name + '.pth'

        tune_features_save_path = 'tune_features/'+ select_pretrained + '/' + select_distance + '/' + file_name + '.npy'
        
        
        
        if select_pretrained=='swim':
        
            model = SwinFeatureExtractor()
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.to(device)
            model.eval()
        elif select_pretrained=='resnet':
            model = ResNetFeatureExtractor()
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.to(device)
            model.eval()
        elif select_pretrained=='ConvNeXt':
            model = ConvNeXtFeatureExtractor()
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            model.to(device)
            model.eval()
        else:
            pass
        
        all_features = []
    
        with torch.no_grad():
            for batch_imgs, _ in dataloader:
                batch_imgs = batch_imgs.to(device)
                batch_feats = model(batch_imgs)  # shape: [batch_size, 512]
                all_features.append(batch_feats.cpu().numpy())
    
        features_array = np.vstack(all_features)  
        
        
        #save_path = os.path.join(features_root, f"{file_name}_features.npy")
        np.save(tune_features_save_path, features_array)
        
    
        print(f"Saved features to: {tune_features_save_path}")
    