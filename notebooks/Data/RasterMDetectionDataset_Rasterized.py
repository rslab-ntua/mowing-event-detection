import json
import numpy as np
import os
import random
import rasterio
import math
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A
import copy
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Recall, Precision
from torchinfo import summary
from albumentations.pytorch.transforms import ToTensorV2
from settings import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


class RasterMowingDataset(Dataset):
    ###if interpolation is true nan_must be false###
    def __init__(self,
                 root_path,
                 labels_path,
                 trsm = None,
                 indices=None,
                 mode = 'normal',
                #interpolation = False,
                nan_free = False,
                max_lenght_to_pad = 70,
                concat_mask = True,
                channels = input_channels,
                num_samples = 10):
        
        super().__init__()
        
        self.labels_path = labels_path
        self.root_path = root_path
        self.nan_free = nan_free
        self.transformation = trsm
        self.mode = mode
        self.indices = indices
        self.max_lenght_to_pad = max_lenght_to_pad
        self.concat_mask = concat_mask
        self.input_channels = channels
        self.num_samples = num_samples

        ####make the db####
        self.mk_db()

    
    def mk_db(self):
        self.data_list = []
        with open(self.labels_path, 'r') as file:
            json_data = file.read()
        
        self.data_files = json.loads(json_data)
        
        for lb in self.data_files:
           

            parcel_var = list(lb.keys())[0]
            class_var = list(lb.values())[0]

            self.data_list.append((os.path.join(self.root_path,class_var, parcel_var)+'.tif',int(class_var)))
        if self.indices is not None:
            self.data_list = [self.data_list[i] for i in self.indices]
        print('Generated db list and has ',len(self.data_list), 'parcels')
                

#'normal' or 'pad' is used to indicate if the data should be padded or not
    def __getitem__(self, idx): 
        print('entered get item')
        path, label = self.data_list[idx][0], self.data_list[idx][1]
        dataset = rasterio.open(path)
        steps = dataset.read()
        steps = np.nan_to_num(steps,nan = 0)
        print('steps shape is',steps.shape)
        '''
        T, x, y = steps.shape
        reshaped_steps = steps.reshape((T,-1))
        #print(reshaped_steps.shape)
        reshaped_steps = np.nanmean(reshaped_steps, axis = 1)
        reshaped_steps = reshaped_steps.reshape(((int(T/self.input_channels),self.input_channels)))
        if self.nan_free:
            reshaped_steps = reshaped_steps[~np.isnan(reshaped_steps)]
            reshaped_steps = reshaped_steps.reshape(-1, 1)
            #print(reshaped_steps.shape)
        if self.mode=="normal":
            data = reshaped_steps#[:,indices,:]
        else:
            data = np.pad(reshaped_steps,((0, self.max_lenght_to_pad-len(reshaped_steps)), (0, 0)),mode = 'constant')
            
            
        
        if self.transformation is not None:
            data = np.transpose(data,(0,1))
            data = self.transformation(image=data)['image']
        '''
        data = steps
        return data,label
        
        
    def __len__(self):
        return len(self.data_list)
    
    def get_labels(self):
        print('entered get labels')
        return [entry[1] for entry in self.data_list]


