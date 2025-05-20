import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning as pl
import copy
import math

from torch import nn
from torchinfo import summary
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Recall, Precision
from torchinfo import summary
from settings import *
from Data.MowingDetectionDataset_Parcelized import *
from System.pos_enc import *
from System.helpers_Tr import *
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

print(mlp1_size)

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:20:16 2021

@author: bjorn

script for transformer model
"""
#import torch
#import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#from model_utils import PositionalEncoding, SelfAttentionPooling

class TransformerModel(nn.Module):

    def __init__(self, d_model=64, nhead=4, dim_feedforward=1024, nlayers=6, n_conv_layers=2, n_class=2, dropout=0.2, dropout_other=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.n_class = n_class
        self.n_conv_layers = n_conv_layers
        self.relu = torch.nn.ReLU()
        self.pos_encoder = PositionalEncoding(64, dropout)
        #self.pos_encoder2 = PositionalEncoding(6, dropout)
        self.self_att_pool = SelfAttentionPooling(d_model)
        #self.self_att_pool2 = SelfAttentionPooling(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward, 
                                                 dropout=dropout
                                                 )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.flatten_layer = torch.nn.Flatten()
        # Define linear output layers
        if n_class == 2:
          self.decoder = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                       nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        # else:
        #   self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(0.1),
        #                                nn.Linear(d_model, d_model), nn.Dropout(0.1), 
        #                                nn.Linear(d_model, n_class))
        if n_class == 2:
          self.decoder2 = nn.Sequential(nn.Linear(d_model, d_model), 
                                       nn.Dropout(dropout_other),
                                      #  nn.Linear(d_model, d_model), 
                                       nn.Linear(d_model, 64))
        # Linear output layer after concat.
        self.fc_out1 = torch.nn.Linear(64, 64)
        self.fc_out2 = torch.nn.Linear(64, 5) # if two classes problem is binary  
        # self.init_weights()
        # Transformer Conv. layers
        self.conv1 = torch.nn.Conv1d(in_channels=5, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
        self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(d_model)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=0.1)
        # self.avg_maxpool = nn.AdaptiveAvgPool2d((64, 64))
        # RRI layers
        self.conv1_rri = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.conv2_rri = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3) 

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):      
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # size input: [batch, sequence, embedding dim.]
        # src = self.pos_encoder(src) 
        src=src.squeeze(dim=1)
        src = src.permute(0,2,1)
        #print('initial src shape:', src.shape)
        #src = src.view(-1, 1, src.shape[1]) # Resize to --> [batch, input_channels, signal_length]
        src = self.relu(self.conv1(src))
        src = self.relu(self.conv2(src))
        # src = self.maxpool(self.relu(src))
        #print('src shape after conv1:', src.shape)
        for i in range(self.n_conv_layers):
          src = self.relu(self.conv(src))
          src = self.maxpool(src)
        #print('src shape after max pooling:',src.shape)
        # src = self.maxpool(self.relu(src))
        src = self.pos_encoder(src)   
        # print(src.shape) # [batch, embedding, sequence]
        src = src.permute(2,0,1) # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        # print('src shape:', src.shape)
        output = self.transformer_encoder(src) # output: [sequence, batch, embedding dim.], (ex. [3000, 5, 512])
        # print('output shape 1:', output.shape)
        # output = self.avg_maxpool(output)
        # output = torch.mean(output, dim=0) # take mean of sequence dim., output: [batch, embedding dim.] 
        output = output.permute(1,0,2)
        output = self.self_att_pool(output)
        # print('output shape 2:', output.shape)
        logits = self.decoder(output) # output: [batch, n_class]
        # print('output shape 3:', logits.shape)
        # output_softmax = nn.functional.softmax(logits, dim=1) # get prob. of logits dim.  # F.log_softmax(output, dim=0)
        # output = torch.sigmoid(output)
        # RRI layers
        '''
        src2 = src2.view(-1, 1, src2.shape[1]) # Resize to --> [batch, input_channels, signal_length]
        src2 = self.relu(self.conv1_rri(src2))
        src2 = self.relu(self.conv2_rri(src2))
        src2 = self.pos_encoder2(src2)  
        src2 = src2.permute(2,0,1) # reshape from [batch, embedding dim., sequnce] --> [sequence, batch, embedding dim.]
        output2 = self.transformer_encoder2(src2)
        output2 = output2.permute(1,0,2)
        output2 = self.self_att_pool2(output2)
        logits2 = self.decoder2(output2) # output: [batch, n_class]
        logits_concat = torch.cat((logits, logits2), dim=1)
        # Linear output layer after concat.
        '''
        xc = self.flatten_layer(logits)
        #print('shape after flatten', xc.shape)
        xc = self.fc_out2(self.dropout(self.relu(self.fc_out1(xc)))) 

        return xc




class ECGTRMODEL(pl.LightningModule):
    def __init__(self, lr=learning_rate, num_classes = number_of_classes):
      
        super(ECGTRMODEL, self).__init__()
        # input to MLP (in this model is 45)
        # Cross Entropy Loss
        self.CEL = nn.CrossEntropyLoss()
        # learnig rate
        self.lr = lr
        # resnet model
        self.tr_moddel_dec = TransformerModel()
        
        ###metrics###
        # train acc
        self.train_accuracy = Accuracy(task="multiclass",average = 'weighted', num_classes=num_classes)
        # val acc macro/micro
        self.val_accuracy_macro = Accuracy(task="multiclass",average = 'macro', num_classes=num_classes)
        self.val_accuracy_micro = Accuracy(task="multiclass",average = 'micro', num_classes=num_classes)
        # val f1 macro/micro
        self.val_F1_macro = F1Score(task="multiclass",average = 'macro', num_classes=num_classes)
        self.val_F1_micro = F1Score(task="multiclass",average = 'micro', num_classes=num_classes)
        # val recall macro/micro
        self.val_Recall_macro = Recall(task="multiclass",average = 'macro', num_classes=num_classes)
        self.val_Recall_micro = Recall(task="multiclass",average = 'micro', num_classes=num_classes)
        # val prec macro/micro
        self.val_Prec_macro = Precision(task="multiclass",average = 'macro', num_classes=num_classes)
        self.val_Prec_micro = Precision(task="multiclass",average = 'micro', num_classes=num_classes)
        # test acc macro/micro
        self.test_accuracy_macro = Accuracy(task="multiclass",average = 'macro', num_classes=num_classes)
        
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)     

        # val f1 macro/micro
        self.test_F1_macro = F1Score(task="multiclass",average = 'macro', num_classes=num_classes)

        # val recall macro/micro
        self.test_Recall_macro = Recall(task="multiclass",average = 'macro', num_classes=num_classes)

        # val prec macro/micro
        self.test_Prec_macro = Precision(task="multiclass",average = 'macro', num_classes=num_classes)   


        self.save_hyperparameters()       
        

    def forward(self, x: torch.Tensor):
        out = self.tr_moddel_dec(x)
        return out
      
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        #print(logits.shape)
        #print(y.shape)
        loss = self.CEL(logits, y)
        #print(loss)
        self.log("loss/train", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.train_accuracy(logits, y)
        self.log("accuracy/train", self.train_accuracy, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)

        loss = self.CEL(logits, y)
        self.log("loss/val", loss, prog_bar=True, on_epoch=True, on_step=False)
        # val acc macro/micro
        self.val_accuracy_macro(logits, y)
        self.log("accuracy_macro/val", self.val_accuracy_macro, prog_bar=True, on_epoch=True, on_step=False)
        self.val_accuracy_micro(logits, y)
        self.log("accuracy_micro/val", self.val_accuracy_micro, prog_bar=True, on_epoch=True, on_step=False)
        # val f1 macro/micro
        self.val_F1_macro(logits, y)
        self.log("val_F1_macro/val", self.val_F1_macro, prog_bar=True, on_epoch=True, on_step=False)
        self.val_F1_micro(logits, y)
        self.log("val_F1_micro/val", self.val_F1_micro, prog_bar=True, on_epoch=True, on_step=False)
        # val recall macro/micro
        self.val_Recall_macro(logits, y)
        self.log("val_Recall_macro/val", self.val_Recall_macro, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Recall_micro(logits, y)
        self.log("val_Recall_micro/val", self.val_Recall_micro, prog_bar=True, on_epoch=True, on_step=False)
        # val prec macro/micro
        self.val_Prec_macro(logits, y)
        self.log("val_Prec_macro/val", self.val_Prec_macro, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Prec_micro(logits, y)
        self.log("val_Prec_micro/val", self.val_Prec_micro, prog_bar=True, on_epoch=True, on_step=False) 
        
        
        self.val_confusion_matrix(logits, y)
    def test_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)

        loss = self.CEL(logits, y)
        self.log("loss/test", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.test_accuracy_macro(logits, y)
        self.log("accuracy/test", self.test_accuracy_macro, prog_bar=True, on_epoch=True, on_step=False)

        self.test_F1_macro(logits, y)
        self.log("test_F1_macro/test", self.test_F1_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val recall macro/micro
        self.test_Recall_macro(logits, y)
        self.log("test_Recall_macro/test", self.test_Recall_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val prec macro/micro
        self.test_Prec_macro(logits, y)
        self.log("test_Prec_macro/test", self.test_Prec_macro, prog_bar=True, on_epoch=True, on_step=False)
        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 10)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }