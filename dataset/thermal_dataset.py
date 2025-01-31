import torch 
import numpy as np

class thermal_dataset(torch.utils.data.Dataset):
  def __init__(self, data, lookback, device):
    super().__init__()
    X, y = [], []
    #print(data)
    #print(lookback)
    #print(len(data[0])-lookback)
    for i in range(lookback, len(data[0])):
      feature = data[0][i-lookback:i]
      target = data[1][i:i+2]
      X.append(feature)
      y.append(target)
    self.X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    self.y = torch.tensor(np.array(y), dtype=torch.float32).to(device)

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return len(self.X)