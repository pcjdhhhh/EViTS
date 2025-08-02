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
warnings.filterwarnings("ignore")

root_dir = 'output_images'  
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "swim_transformer"
features_root = os.path.join(os.path.dirname(root_dir), f"features_{model_name}")
os.makedirs(features_root, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),       
    transforms.ToTensor(),            
    transforms.Normalize(              
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225])
])




model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
model.eval()
model.to(device)





for dataset_name in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_name)
    

    print(f"Processing dataset: {dataset_name}")

    dataset = ImageFolderDataset(dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []

    with torch.no_grad():
        for batch_imgs, _ in dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_feats = model(batch_imgs)  # shape: [batch_size, 512]
            all_features.append(batch_feats.cpu().numpy())

    features_array = np.vstack(all_features)  
    
    
    save_path = os.path.join(features_root, f"{dataset_name}_features.npy")
    np.save(save_path, features_array)
    

    print(f"Saved features to: {save_path}")
    