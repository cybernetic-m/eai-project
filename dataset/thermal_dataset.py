import torch 
import numpy as np

class thermal_dataset(torch.utils.data.Dataset):
  def __init__(self, data, lookback, device):
    super().__init__()
    X, y = [], []
    #print(data)
    #print(lookback)
    #print(len(data[0])-lookback)
    for i in range(len(data[0])-lookback):
      feature = data[0][i:i+lookback]
      target = data[1][i+1]
      X.append(feature)
      y.append(target)
    self.X = torch.tensor(np.array(X)).to(device)
    self.y = torch.tensor(np.array(y)).to(device)

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return len(self.X)