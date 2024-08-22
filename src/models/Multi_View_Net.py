import torch.nn as nn
import torch

class Multi_View_NN(nn.Module):
  '''
  function for building Multi View Neural network 
  
  args:
    input: int
    hidden: int
  '''
  def __init__(self, txt_input:int=768, ft_input:int=7, fusion_input:int=768, hidden:int=1024):
    super().__init__()
    self.text_block = nn.Sequential(
        nn.Linear(txt_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),   
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),    
        )
    self.feature_block = nn.Sequential(
        nn.Linear(ft_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),   
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.fusion_block = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),   
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.fusion_res_block = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.fusion_block_2 = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.cls_block = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.res_block = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.cls_block_2 = nn.Sequential(
        nn.Linear(fusion_input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,fusion_input),   
        )
    self.fc_layer = nn.Linear(fusion_input,1)
    
  def forward(self, x:list):
    x_txt = self.text_block(x[:,7:])
    x_ft = self.feature_block(x[:,:7])
    x = torch.stack([x_txt,x_ft],dim=1)
    x = self.fusion_block(x)
    x_ = self.fusion_res_block(x)
    x = self.fusion_block_2(x+x_)
    x = self.cls_block(x[:,0])
    x_ = self.res_block(x)
    x = self.cls_block_2(x+x_)
    x = self.fc_layer(x)
    return x