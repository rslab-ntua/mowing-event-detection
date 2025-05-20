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


class ParcelMowingDataset(Dataset):
    ###if interpolation is true nan_must be false###
    def __init__(self,
                 root_path,
                 labels_path,
                 trsm = None,
                 indices=None,
                 mode = 'normal',
                nan_free = False,
                max_lenght_to_pad = 70,
                extra_features = True,
                channels = input_channels,
                num_samples = 10,
                interpolate = None):
        
        super().__init__()
        
        self.labels_path = labels_path
        self.root_path = root_path
        self.nan_free = nan_free
        self.transformation = trsm
        self.mode = mode
        self.indices = indices
        self.max_lenght_to_pad = max_lenght_to_pad
        self.input_channels = channels
        self.num_samples = num_samples
        self.interpolate = interpolate
        self.extra_features = extra_features

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
        ###in case we want to use part a subset of a dataset###
        if self.indices is not None:
            self.data_list = [self.data_list[i] for i in self.indices]
        print('Generated db list and has ',len(self.data_list), 'parcels')
                

#'normal' or 'pad' is used to indicate if the data should be padded or not
    def __getitem__(self, idx):     
        path, label = self.data_list[idx][0], self.data_list[idx][1]
        dataset = rasterio.open(path)
        steps = dataset.read()
        #print(steps.shape)
        T, x, y = steps.shape
        reshaped_steps = steps.reshape((T,-1))
        # take the mean of each timestep
        reshaped_steps = np.nanmean(reshaped_steps, axis = 1)
        reshaped_steps = reshaped_steps.reshape(((int(T/self.input_channels),self.input_channels)))
        # if we want to take the nan free ts pass True
        if self.nan_free:
            reshaped_steps = reshaped_steps[~np.isnan(reshaped_steps)]
            reshaped_steps = reshaped_steps.reshape(-1, 1)
            
        #### interpolate in forward direction ###########
        #### and then remove anynan values at the start##
        if self.interpolate:
            # Create a Pandas DataFrame
            df = pd.DataFrame(reshaped_steps)
            df.interpolate(method='linear', axis=0, inplace = True, limit_direction='forward')
            data = df.to_numpy()
            ###check if there are any nans at the beginning and remove###
            reshaped_steps = data[~np.isnan(data)]
            reshaped_steps = reshaped_steps.reshape(-1, 1)
            #print('data after nan free ',reshaped_steps.shape)
        data = reshaped_steps
        #print('before extra')
        if self.extra_features:
            #print('before max',data.shape)
            newM = self.diff_from_max(data)
            #print(data)
            newm = self.diff_from_min(data)
            new_mean = self.diff_from_mean(data)
            #print('before diff',data.shape)
            diff = self.diff_from_next(data)
            #print('before concat')
            data = np.concatenate((newM,newm,new_mean,diff,data), axis = 1)
            data = np.float32(data)
        ### pass normal to use unpadded sequence############
        ### else sewuence will be padded intil max length###
        if self.mode=="normal":
            data = data#[:,indices,:]
        else:
            data = np.pad(data,((0, self.max_lenght_to_pad-len(data)), (0, 0)),mode = 'constant')
            
        ###perform the transformations###
        if self.transformation is not None:
            data = np.transpose(data,(0,1))
            data = self.transformation(image=data)['image']
            
        return data,label
        
        
    def __len__(self):
        return len(self.data_list)
    
    def get_labels(self):
        print('entered get labels')
        return [entry[1] for entry in self.data_list]

    def diff_from_max(self,array):
        # find max value
        max_value = np.max(array, axis=0)
        # create an empty array to contain max 
        max_full = np.empty_like(array)
        max_full.fill(max_value)
        # difference from max value for every step
        to_concat = max_full-array
        return to_concat

    def diff_from_min(self,array):
        min_value = np.min(array, axis=0)
        min_full = np.empty_like(array)
        min_full.fill(min_value)
        to_concat = array-min_full
        return to_concat

    def diff_from_mean(self,array):
        mean_value = np.mean(array, axis=0)
        mean_full = np.empty_like(array)
        mean_full.fill(mean_value)
        to_concat = array-mean_full
        return to_concat

    def diff_from_next(self,array):
        differences = np.diff(array, axis=0)
        zero_array = np.zeros((1, 1))
        differences = np.vstack((differences, zero_array))
        return differences
