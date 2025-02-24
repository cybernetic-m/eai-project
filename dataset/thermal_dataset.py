import torch 
import numpy as np
from torch.utils.data import Dataset

class thermal_dataset(Dataset):
    def __init__(self, data, lookback, device):
        super().__init__()
        X, y = [], []
        for i in range(lookback, len(data[0])):
            feature = data[0][i-lookback:i]
            target = data[1][i-1:i+1]
            if target.shape != (2, 3):
                continue
            y.append(target)
            X.append(feature)
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.device = device

    def __getitem__(self, index):
        x = torch.tensor(self.X[index], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y[index], dtype=torch.float32).to(self.device)
        return x, y

    def __len__(self):
        return len(self.X)