# -*- coding: utf-8 -*-


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

class DistanceSupervisedDataset(Dataset):
    
    def __init__(self,image_dir,image_pairs,DTW_pairs_dis,MSM_pairs_dis,TWED_pairs_dis,select_distance,transform=None):
        """
        image_pairs: List of (img_path_1, img_path_2)
        dtw_distances: List of float distances
        """
        self.image_dir = image_dir
        self.pairs = image_pairs
        self.transform = transform
        if select_distance=='DTW':
            self.dis = DTW_pairs_dis
        elif select_distance=='MSM':
            self.dis = MSM_pairs_dis
        else:
            self.dis = TWED_pairs_dis

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        path1 = self.image_dir + '/' + self.pairs[idx][0]
        path2 = self.image_dir + '/' + self.pairs[idx][1]
        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')
        d = torch.tensor(self.dis[idx], dtype=torch.float)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, d


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name