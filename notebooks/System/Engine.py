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
from helpers.various import *
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchscan import summary
from System.Engine_Tr import *
print('Engine loaded')



###plain MLP for multivariate approach###

class MLP_MV(nn.Module):
    def __init__(self,in_size,hid1,number_of_classes):
        super(MLP_MV,self).__init__()
        

        self.in_size = in_size
        self.hid1 = hid1
        self.hid2 = 2*hid1
        #self.hid2 = hid1
        self.number_of_classes = number_of_classes
        
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.in_size,self.hid1)
        self.fc2 = nn.Linear(self.hid1,self.hid2)
        self.fc3 = nn.Linear(self.hid2,self.number_of_classes)
        self.relu = nn.ReLU()
        
        

        
        
    def forward(self, x):
        
        x = self.flatten(x)
        #print(x.shape)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

class Model_MLP_MV(pl.LightningModule):
    def __init__(self, lr=learning_rate, in_size = in_size, hid1 = hid1, num_classes = number_of_classes):
      
        super(Model_MLP_MV, self).__init__()
        
        self.in_size = in_size
        self.hid1 = hid1
        self.lr = lr
        self.Floss = FocalLoss()
        self.CEL = nn.CrossEntropyLoss()
        self.Mlp = MLP_MV(self.in_size,self.hid1,num_classes)
        
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
      
        ###save various###
        self.save_hyperparameters()
        
        

    def forward(self, x: torch.Tensor):
        
        out = self.Mlp(x)
        
        return out
      
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        #print('logits shape is ',logits.shape)
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
        self.log('text_message', 90)
        
        self.val_confusion_matrix(logits, y)
    def test_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)

        loss = self.CEL(logits, y)
        self.log("loss/test", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.test_accuracy_macro(logits, y)
        self.log("accuracy/test", self.test_accuracy_macro, prog_bar=True, on_epoch=True, on_step=False)

        self.test_F1_macro(logits, y)
        self.log("val_F1_macro/val", self.test_F1_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val recall macro/micro
        self.test_Recall_macro(logits, y)
        self.log("test_Recall_macro/val", self.test_Recall_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val prec macro/micro
        self.test_Prec_macro(logits, y)
        self.log("test_Prec_macro/val", self.test_Prec_macro, prog_bar=True, on_epoch=True, on_step=False)

        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 20)
        #scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }







###linear container###

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        
        x = self.linear(x)  
        #print(x.dim())
        if x.dim() == 3:  
            
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # (B, C)
            x = self.norm(x)
        return self.activation(x)
    





class MlpContainer(nn.Module):
    def __init__(self,  mlp1=mlp1_size):
        super(MlpContainer,self).__init__()

        self.mlp1_dim = copy.deepcopy(mlp1)
        
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(LinearLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)
        

        
        
    def forward(self, x):
        batch, temp = x.shape[:2]
        x=x.squeeze(dim=1)
        #print('mlp cont x',x.shape)
        pad_mask = (x == 0)
        #print('mlp cont pad mask',pad_mask.shape)
        pad_mask = pad_mask.squeeze(axis=-1)
        #print('mlp cont pad mask squeeze',pad_mask.shape)
        #print(x.shape)
        out = x.view(batch * temp, *x.shape[1:])
        #print('mlp cont out',out.shape)
        out = torch.nan_to_num(out, nan = 0)
        out = self.mlp1(out)
        #print('out after mlp1',out.shape)  

        return out,pad_mask

    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    
###model where each of the steps is encoded using an MLP, then passed through a transformer encoder###   
class Model(pl.LightningModule):
    def __init__(self, embedding_dim = embed_dim,
                 dropout=dropout_val,
                 mlp_size=size_of_mlp_tr,
                 num_trans_layers=tr_layers,
                 num_heads=n_heads,
                 num_classes = number_of_classes,
                 mlp_ratio=3,
                 n_layers=number_of_layers,
                 lr=learning_rate):
      
        super(Model, self).__init__()
        
        self.lr = lr
        self.Floss = FocalLoss()
        self.CEL = nn.CrossEntropyLoss()
        self.MplC = MlpContainer()
        
        ###create the class token###
        self.class_token = nn.Parameter(torch.rand(1, embedding_dim))
        ###create position embedding###
        self.pos_embed = PositionalEncoding(49, embedding_dim)

        ###create encoder layer###        
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead = 4, 
                                                      batch_first=True)
        ###create the encoder###
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        ###the classifier###
        self.classifier = nn.Linear(embedding_dim, num_classes)
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
        

    def forward(self, x: torch.Tensor):
        ###pass through the MLP###
        tokens, pad_mask = self.MplC(x)
        #print(pad_mask.shape)
        ###create the token step for the mask, set it to False so it is attended###
        token_step = torch.full((pad_mask.size(0), 1,5), False, dtype=torch.bool, device=self.device)
        #print('token step is of shape',token_step.shape)
        #print('pad_mask is of shape',pad_mask.shape)
        ###concatenate with the rest###
        #pad_mask = torch.cat((token_step, pad_mask), dim=1).to(self.device)#.cuda()

        ###final tokens with class token###
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        ###add position embedding###
        out = self.pos_embed(tokens)
        ###pass through the transformer###
        #out = self.transformer_encoder(out,src_key_padding_mask=pad_mask)
        out = self.transformer_encoder(out)
        #print('token out', out.shape)
        ###finally classify using the token###
        out = self.classifier(out[:,0])
        return out
      
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        #print('logits shape is ',logits.shape)
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
        self.log("val_F1_macro/val", self.test_F1_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val recall macro/micro
        self.test_Recall_macro(logits, y)
        self.log("test_Recall_macro/val", self.test_Recall_macro, prog_bar=True, on_epoch=True, on_step=False)

        # val prec macro/micro
        self.test_Prec_macro(logits, y)
        self.log("test_Prec_macro/val", self.test_Prec_macro, prog_bar=True, on_epoch=True, on_step=False)

        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 20)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }

    
    
class ModelMLP(pl.LightningModule):
    def __init__(self, input_size = 48,
                 hidden_size = 128,
                 num_classes = number_of_classes,
                 lr=learning_rate):
      
        super(ModelMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lr = lr
        self.Floss = FocalLoss()
        self.CEL = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes)
        
        self.train_accuracy = Accuracy(task="multiclass",average = 'weighted', num_classes=num_classes)
        
        self.val_accuracy_w = Accuracy(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_accuracy_m = Accuracy(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_F1_w = F1Score(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_F1_m = F1Score(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_Recall_w = Recall(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_Recall_m = Recall(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_Prec_w = Precision(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_Prec_m = Precision(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()
        
        

    def forward(self, x: torch.Tensor):
        x=x.squeeze(dim=1)
        #x = x.permute(0,2,1)
        #print(x.shape)
        x = self.flatten(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        #out = F.softmax(out.squeeze(dim=1),dim=1)
        #print(out.shape)
        return out
      
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        #print(logits.shape)

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

        self.val_accuracy_m(logits, y)
        self.log("accuracy_m/val", self.val_accuracy_m, prog_bar=True, on_epoch=True, on_step=False)
        self.val_accuracy_w(logits, y)
        self.log("accuracy_w/val", self.val_accuracy_w, prog_bar=True, on_epoch=True, on_step=False)
        
        self.val_F1_w(logits, y)
        self.log("val_F1_w/val", self.val_F1_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_F1_m(logits, y)
        self.log("val_F1_m/val", self.val_F1_m, prog_bar=True, on_epoch=True, on_step=False)

        self.val_Recall_w(logits, y)
        self.log("val_Recall_w/val", self.val_Recall_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Recall_m(logits, y)
        self.log("val_Recall_m/val", self.val_Recall_m, prog_bar=True, on_epoch=True, on_step=False)
        
        self.val_Prec_w(logits, y)
        self.log("val_Prec_w/val", self.val_Prec_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Prec_m(logits, y)
        self.log("val_Prec_m/val", self.val_Prec_m, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('text_message', 90)
        
        self.val_confusion_matrix(logits, y)
    def test_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)

        loss = self.CEL(logits, y)
        self.log("loss/test", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.val_accuracy_w(logits, y)
        self.log("accuracy/test", self.val_accuracy_w, prog_bar=True, on_epoch=True, on_step=False)

        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }
    

    
    
    
    
class ModelMLPFORMER(pl.LightningModule):
    def __init__(self, embedding_dim = 1,
                 dropout=dropout_val,
                 input_size = 45,
                 hidden_size = 128,
                 num_trans_layers=tr_layers,
                 num_heads=1,
                 num_classes = number_of_classes,
                 mlp_ratio=3,
                 n_layers=number_of_layers,
                 lr=learning_rate):
      
        super(ModelMLPFORMER, self).__init__()
        # input to MLP (in this model is 45)
        self.input_size = input_size
        # hidden size of MLP 
        self.hidden_size = hidden_size
        # number of classes 
        self.num_classes = num_classes
        # learnig rate
        self.lr = lr
        # number of tr5ansformer encoder layers
        self.num_trans_layers=number_of_layers
        # number of heads
        self.num_heads=num_heads
        # embedding dimention to transformer
        self.embedding_dim = embedding_dim
        
        # Cross Entropy Loss
        self.CEL = nn.CrossEntropyLoss()
        # Linear layers 
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        # Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.embedding_dim))
        # position embedding
        self.pos_embed = PositionalEncoding(129, self.embedding_dim)
        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, nhead = 1, 
                                                      
                                                     batch_first=True)
        # Transformer encoder with the above layer (X number of layers)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_trans_layers)
        #  the classifier
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        
        
        
        
        print('em dddim',self.embedding_dim,'enc layers',self.num_trans_layers)
        self.train_accuracy = Accuracy(task="multiclass",average = 'weighted', num_classes=num_classes)
        
        self.val_accuracy_w = Accuracy(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_accuracy_m = Accuracy(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_F1_w = F1Score(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_F1_m = F1Score(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_Recall_w = Recall(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_Recall_m = Recall(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_Prec_w = Precision(task="multiclass",average = 'weighted', num_classes=num_classes)
        self.val_Prec_m = Precision(task="multiclass",average = 'micro', num_classes=num_classes)
        
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()
        
        

    def forward(self, x: torch.Tensor):
        x=x.squeeze(dim=1)
        x = x.permute(0,2,1)
        #print(x.shape)
        
        out = self.fc1(x)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = out.permute(0,2,1)
        #print('tokens are of shape',out.shape)
        #print(tokens)
        out = torch.stack([torch.vstack((self.class_token, out[i])) for i in range(len(out))])
   
        out = self.pos_embed(out)

        out = self.transformer_encoder(out)
        #print('token out pre classifier', out[:,0].shape)
        out = self.classifier(out[:,0])
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

        self.val_accuracy_m(logits, y)
        self.log("accuracy_m/val", self.val_accuracy_m, prog_bar=True, on_epoch=True, on_step=False)
        self.val_accuracy_w(logits, y)
        self.log("accuracy_w/val", self.val_accuracy_w, prog_bar=True, on_epoch=True, on_step=False)
        
        self.val_F1_w(logits, y)
        self.log("val_F1_w/val", self.val_F1_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_F1_m(logits, y)
        self.log("val_F1_m/val", self.val_F1_m, prog_bar=True, on_epoch=True, on_step=False)

        self.val_Recall_w(logits, y)
        self.log("val_Recall_w/val", self.val_Recall_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Recall_m(logits, y)
        self.log("val_Recall_m/val", self.val_Recall_m, prog_bar=True, on_epoch=True, on_step=False)
        
        self.val_Prec_w(logits, y)
        self.log("val_Prec_w/val", self.val_Prec_w, prog_bar=True, on_epoch=True, on_step=False)
        self.val_Prec_m(logits, y)
        self.log("val_Prec_m/val", self.val_Prec_m, prog_bar=True, on_epoch=True, on_step=False) 
        self.log('text_message', 90)
        
        self.val_confusion_matrix(logits, y)
    def test_step(self, batch, batch_idx):
        X, y = batch

        logits = self(X)

        loss = self.CEL(logits, y)
        self.log("loss/test", loss, prog_bar=True, on_epoch=True, on_step=False)

        self.val_accuracy_w(logits, y)
        self.log("accuracy/test", self.val_accuracy_w, prog_bar=True, on_epoch=True, on_step=False)

        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }
    
    
    
    
    
    
    
    
    
import torch
import torch.nn as nn


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=5, inplanes=64, num_classes=5):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=5, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=1)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print(x.shape)
        return x


def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model






class resnet(pl.LightningModule):
    def __init__(self, num_classes = number_of_classes,
                 lr=learning_rate):
      
        super(resnet, self).__init__()

        self.num_classes = num_classes
        # Cross Entropy Loss
        self.CEL = nn.CrossEntropyLoss()
        # learnig rate
        self.lr = lr
        # resnet model
        self.resnet1d = resnet18()
        
        
        
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
        


        # val f1 macro/micro
        self.test_F1_macro = F1Score(task="multiclass",average = 'macro', num_classes=num_classes)

        # val recall macro/micro
        self.test_Recall_macro = Recall(task="multiclass",average = 'macro', num_classes=num_classes)

        # val prec macro/micro
        self.test_Prec_macro = Precision(task="multiclass",average = 'macro', num_classes=num_classes)   


        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)      
        

    def forward(self, x: torch.Tensor):
        
        x=x.squeeze(dim=1)
        x = x.permute(0,2,1)
        #print(x.shape)
        
        out = self.resnet1d(x)
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
        #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 60)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }
    
    
    
    
    
    
class Pretrained_resnet(pl.LightningModule):
    def __init__(self, embedding_dim = 1,
                 dropout=dropout_val,
                 input_size = 45,
                 hidden_size = 128,
                 num_trans_layers=tr_layers,
                 num_heads=1,
                 num_classes = number_of_classes,
                 mlp_ratio=3,
                 n_layers=number_of_layers,
                 lr=learning_rate,
                path_checkpoint = path_to_pretrained_resnet):
      
        super(Pretrained_resnet, self).__init__()
        # input to MLP (in this model is 45)
        # Cross Entropy Loss
        self.CEL = nn.CrossEntropyLoss()
        # learnig rate
        self.lr = lr
        # resnet model

        self.path_checkpoint = path_checkpoint
        
        self.pretrained_model = resnet.load_from_checkpoint(self.path_checkpoint)
        # Freeze all the parameters in the network
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
        # Get the number of input features for the last FC layer

        num_features = self.pretrained_model.resnet1d.fc.in_features
        #print('num features_in',num_features)
        # Define your new FC layer
        # For example, if you want to replace the FC layer with a new one for a different classification task:
        new_fc = nn.Linear(num_features, 5)  # num_classes is the number of classes in your new task

        # Replace the FC layer in the ResNet model
        self.pretrained_model.resnet1d.fc = new_fc
        
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
        self.test_F1_macro = F1Score(task="multiclass",average = 'macro', num_classes=num_classes)
        
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)        
        

    def forward(self, x: torch.Tensor):
        
        x=x.squeeze(dim=1)
        #x = x.permute(0,2,1)
        #print(x.shape)
        
        out = self.pretrained_model(x)
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
        self.log("F1/test", self.test_F1_macro, prog_bar=True, on_epoch=True, on_step=False)
        
        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.8)
        #scheduler = StepLR(optimizer, step_size=50, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }
    
    
    
class Pretrained_ECGTR(pl.LightningModule):
    def __init__(self, embedding_dim = 1,
                 dropout=dropout_val,
                 input_size = 45,
                 hidden_size = 128,
                 num_trans_layers=tr_layers,
                 num_heads=1,
                 num_classes = number_of_classes,
                 mlp_ratio=3,
                 n_layers=number_of_layers,
                 lr=learning_rate,
                path_checkpoint = path_to_pretrained_transformer):
      
        super(Pretrained_ECGTR, self).__init__()
        # input to MLP (in this model is 45)
        # Cross Entropy Loss
        self.CEL = nn.CrossEntropyLoss()
        # learnig rate
        self.lr = lr
        # resnet model

        self.path_checkpoint = path_checkpoint
        
        self.pretrained_model = ECGTRMODEL.load_from_checkpoint(self.path_checkpoint)
        self.freeze_layers(self.pretrained_model, 'tr_moddel_dec.decoder')

        
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
        

    def forward(self, x: torch.Tensor):
        
        x=x.squeeze(dim=1)
        #x = x.permute(0,2,1)
        #print(x.shape)
        
        out = self.pretrained_model(x)
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

        self.test_confusion_matrix(logits, y)
        '''
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    '''    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = ReduceLROnPlateau(optimizer,'min', patience=3, factor = 0.9)
        #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 20)
        scheduler = StepLR(optimizer, step_size=70, gamma=0.9)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'loss/val'
       }
    
    
    def freeze_layers(self,model, freeze_until_layer):
    # Flag to indicate whether we have passed the freeze point
        freeze = False

        for name, param in model.named_parameters():
            #print(name)
            if freeze_until_layer in name:
                freeze = True

            if freeze:
                param.requires_grad = True
            else:
                param.requires_grad = False