import torch.nn as nn
from datetime import datetime
import torch
import os
import sys

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(modules_path)

from blocks import lstm

class lstm_only(nn.Module):
  def __init__(self, hidden_dim, input_dim, output_dim):
    super(lstm_only, self).__init__()
    self.extractor = lstm(hidden_dim, input_dim)
    self.out = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    #print(x.shape)
    h, o = self.extractor(x)
    o = o[:, -1, :]
    x = self.out(o)
    return x
  
  def save(self):

    # Get current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Create the directory of results
    dir_path = 'results/training_' + current_time # path of type 'results/training_2024-12-22_14
    os.makedirs(dir_path, exist_ok=True) # Create the directory

    save_name = 'model.pt' # Model name
    save_path = os.path.join(dir_path, save_name) # path of type '/training_2024-12-22_14-57/model.pt
    torch.save(self.state_dict(), save_path) # Save the model
    print(f'Model saved to {save_path}')
    return dir_path
  
  def load(self, path):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    self.load_state_dict(state_dict)
    print("loaded:", path)