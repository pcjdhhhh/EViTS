# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from data import *
import os
import warnings
from tool import *
warnings.filterwarnings("ignore")

datasets_name = get_name()  

#validation_datasets = validation_datasets[-10:]

print('start')
validation_datasets = ['random_walk','gaussin','mixed_sinx']

plt.style.use('grayscale')  


output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

for dataset_file in validation_datasets:
    save_dir = os.path.join(output_folder, dataset_file)
    os.makedirs(save_dir, exist_ok=True)
    
    data = get_data(dataset_file)
    data = data[0:10000,:]
    global_counter = 0
    
    for idx,time_series in enumerate(data):
        plt.figure(figsize=(4, 2))
        plt.plot(time_series, linewidth=0.5)
        plt.axis('off')
        img_name = f'seq_{global_counter:06d}.png'
        
        img_path = os.path.join(save_dir, img_name)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        global_counter += 1
    print(f'Dataset "{dataset_file}" done → {save_dir}')
    
    
    

