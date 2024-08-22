import torch.nn as nn

class ANN(nn.Module):
  '''
  function for building Neural network 
  
  args:
    input: int
    hidden: int
  '''
  def __init__(self, input:int=775, hidden:int=2048):
    super().__init__()
    self.main_block = nn.Sequential(
        nn.Linear(input,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden),   
        nn.PReLU(),
        nn.Linear(hidden,hidden//4),   
        )
    self.res_block_1 = nn.Sequential(
        nn.Linear(hidden//4,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden//4),   
        )
    self.res_block_2 = nn.Sequential(
        nn.Linear(hidden//4,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,hidden//4),   
        )
    self.fc_block = nn.Sequential(
        nn.Linear(hidden//4,hidden),     
        nn.PReLU(),
        nn.Linear(hidden,1),   
        )
    
  def forward(self, x:list):
    x = self.main_block(x)
    x_1 = self.res_block_1(x)
    x_2 = self.res_block_2(x-x_1)
    x = self.fc_block(x-x_1-x_2)
    return x