import torch.nn as nn

class ANN(nn.Module):
  '''
  function for building Neural network 
  
  args:
    input: int
    hidden: int
  '''
  def __init__(self, input:int=775, hidden:int=1024):
    super().__init__()
    self.linear_stack = nn.Sequential(
        nn.Linear(input,hidden),      #(18,32)
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden,hidden*2),   #(32,64)
        nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,hidden*2), #(64,128)
        nn.ReLU(),
        #nn.Dropout(0.3),     
        #nn.Linear(hidden*4,hidden*4), #(128,128)
        #nn.ReLU(),
        #nn.Dropout(0.3),
        nn.Linear(hidden*2,1),        #(128,1)
        nn.ReLU()
        )
    
  def forward(self, x:list):
    x = self.linear_stack(x)
    return x