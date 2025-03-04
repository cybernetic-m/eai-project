import torch.nn as nn
from datetime import datetime
import torch
import os
import sys

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(modules_path)

from blocks import lstm, rnn
from ensemble_model import ensemble_model

class complete_model(nn.Module):
  def __init__(self, hidden_dim, input_dim, model_dict, device, num_layers=1, mode='auto-weighted', extractor_type='lstm'):
    super(complete_model, self).__init__()

    # Define the feature extractors LSTM and RNN that extract the features 
    # (hidden state) to send to the ensemble model
    if extractor_type == 'lstm':
      self.extractor = lstm(feature_dim=hidden_dim, input_dim=input_dim, num_layers=num_layers).to(device)
    elif extractor_type == 'rnn':
      self.extractor = rnn(hidden_dim, input_dim, num_layers=num_layers).to(device)
    self.ensemble = ensemble_model(model_dict, device, mode=mode)
    # Get current timestamp for the save method
    self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
  
  def get_models(self):
    return self.ensemble.get_models()

  def forward(self, x, y_true):

    h, o = self.extractor(x) # Take the hidden state as features
    #h = h.permute(0,2,1)
    out = self.ensemble(h, y_true)
    return out
  
  def save(self):

    # Create the directory of results
    dir_path = 'results/training_' + self.current_time # path of type 'results/training_2024-12-22_14
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

'''
if __name__ == '__main__' :

  model_dict = {'mlp': {'layer_dim_list': [3,40,50,3]},  
                'ARIMA': {'p': 2, 'd': 0, 'q': 2, 'ps': 1, 'ds': 1, 'qs': 1, 's': 1},
                'linear_regressor': {'in_features': 3, 'out_features': 3}
                }
  #model = ensemble_model(model_dict, mode='auto-weighted')

  if torch.cuda.is_available():
      device = "cuda"
  elif torch.backends.mps.is_available():
      device = "mps"
  else:
      device = "cpu"

  model = complete_model(3, 4, model_dict, device, mode='auto-weighted', extractor_type='rnn')
  t = torch.rand(8, 5, 4).to(device) # Remember to pass from 4 values of temperature to 3 values of features and from 5 -> 1
  y_true = torch.rand(8, 1, 3).to(device)
  out = model(t, y_true)
  #print(out.shape)
'''