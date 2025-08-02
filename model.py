# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import timm

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_tiny'):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.backbone(x)

class SwinFeatureExtractor(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.backbone(x)
    
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224'):
        super().__init__()
       
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')

    def forward(self, x):
        
        return self.backbone(x)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50'):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')

    def forward(self, x):
        
        return self.backbone(x)