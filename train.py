import torch
import torch.nn as nn
import torchmetrics
import os
import json

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional
from tqdm.auto import tqdm

from src.dataset.preprocess import preprocess
from src.util.dataset import CustomDataset
from src.models.ANN import ANN
from src.util.metrics import metrics

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_file):
    script_dir = os.getcwd()  # 현재 스크립트 파일의 디렉토리 경로
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(
    model:nn.Module,
    criterion:callable,
    optimizer:torch.optim.Optimizer,
    data_loader:DataLoader,
    device:str
) -> float:
    '''
    train one epochs
    
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

def validation(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''validation
  
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

def evaluation(
    model:nn.Module,
    data_loader:DataLoader,
    device:str
):
    '''evaluate
  
    Args:
        model: model
        data_loader: data loader
        device: device
    '''
    model.eval()
    pred = []
    with torch.inference_mode():
        for x, _ in data_loader:
            x = x.to(device)
            out = model(x)
            pred.append(out.detach().cpu().numpy())
    return pred

def run(config):
    # train data 전처리
    X,y = preprocess('data/1.Training/라벨링데이터/')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    ds_train = CustomDataset(X_train, y_train)
    ds_val = CustomDataset(X_val, y_val)
    dl_train = DataLoader(ds_train, batch_size=config['batch_size'], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=config['batch_size'])
    # test data 전처리
    X,y = preprocess('data/2.Validation/라벨링데이터/')
    ds_test = CustomDataset(X,y)
    dl_test = DataLoader(ds_test, batch_size=config['batch_size'])
    
    history = {
    'loss':[],
    'val_loss':[],
    'lr':[]
    }
    
    model = ANN(X_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    print(model)
    pbar = range(config['epochs'])
    pbar = tqdm(pbar)
        
    print("Learning Start!")
    for _ in pbar:
        loss = train(model, nn.MSELoss(), optimizer, dl_train, device)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['loss'].append(loss) 
        val_loss = validation(model, nn.MSELoss(), dl_val, device)
        pbar.set_postfix(trn_loss=loss, val_loss=val_loss)
            
    print("Done!")
    torch.save(model.state_dict(), config['name']+'.pth')

    pred = evaluation(model, dl_test, device)
    
    metric_score = metrics(y, pred)
    
    print(metric_score)
            
    
    
    return

if __name__ == "__main__":
    config_file = "src/configs/config.json"
    config = load_config(config_file)
    run(config)