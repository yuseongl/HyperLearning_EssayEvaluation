import torch
import torch.nn as nn
import torchmetrics
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from tqdm.auto import tqdm

from src.dataset.preprocess import preprocess
from src.util.dataset import CustomDataset
from src.models.ANN import ANN

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(777)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backend.mps.is_available() else 'cpu'

def train(
    model:nn.Module,
    criterion:callable,
    optimizer:torch.optim.Optimizer,
    data_loader:DataLoader,
    device:str
) -> float:
    '''
    train function
    
    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    '''
    model.train()
    total_loss = 0.
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(data_loader)

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss,correct = 0.,0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        output = torch.round(output)
        metric.update_state(output, y)

  total_loss = total_loss/len(data_loader.dataset)
  return total_loss 

def run():
    
    X,y = preprocess('data/1.Training/라벨링데이터/')
    #X,y = preprocess('data/2.Validation/라벨링데이터/')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ds_train = CustomDataset(X_train, y_train)
    ds_val = CustomDataset(X_val, y_val)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=32)
    
    history = {
    'loss':[],
    'val_loss':[],
    'lr':[]
    }
    
    model = ANN(X_train.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    if args.train:
        pbar = range(args.epochs)
        if args.pbar:
            pbar = tqdm(pbar)
        
        print("Learning Start!")
        for _ in pbar:
            loss = train(model, RMSELoss(), optimizer, dl, device)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['loss'].append(loss) 
            pbar.set_postfix(trn_loss=loss)
              
        print("Done!")
        torch.save(model.state_dict(), args.output+args.name+'.pth')
        
        
        
        model = ANN(X_trn.shape[-1]).to(device)
        model.load_state_dict(torch.load(args.output+args.name+'.pth'))
        model.eval()
        
        pred = []
        with torch.inference_mode():
            for x in dl_val:
                x = x[0].to(device)
                out = model(x)
                pred.append(out.detach().cpu().numpy())
    
    
    return

if __name__ == "__main__":
  run()