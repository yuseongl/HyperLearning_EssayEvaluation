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
    self.linear_stack = nn.Sequential(
        nn.Linear(input,hidden),      #(18,32)
        nn.LeakyReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden,hidden),   #(32,64)
        nn.LeakyReLU(),
        nn.Linear(hidden,1),        #(128,1)
        )
    
  def forward(self, x:list):
    x = self.linear_stack(x)
    return x