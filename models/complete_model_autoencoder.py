import torch.nn as nn
from datetime import datetime
import torch
import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(modules_path)

from autoencoder import conv_autoencoder, lstm_autoencoder
from ensemble_model import ensemble_model

class complete_model_autoencoder(nn.Module):
  def __init__(self, 
               model_dict, 
               device, 
               autoencoder_dim, 
               timesteps,
               mean=(torch.tensor([1,1,1,1]),torch.tensor([1,1,1])),
               std=(torch.tensor([0,0,0,0]),torch.tensor([0,0,0])),
               max_val=(torch.tensor([1]),torch.tensor([1])),
               min_val=(torch.tensor([-1]),torch.tensor([-1])),
               norm = "Not",
               pooling_kernel_size = 2, 
               padding = "same", 
               pooling = "max", 
               scale_factor = 2, 
               upsample_mode = "linear", 
               mode='auto-weighted',
               dropout = 0.0,
               extractor_type="conv",
               heterogeneous = False
               ):
    
    super(complete_model_autoencoder, self).__init__()

    # Define the feature extractor as an autoencoder from which we take
    # the z_merged latent space
    if extractor_type == 'conv':
      self.extractor = conv_autoencoder(in_kern_out=autoencoder_dim, 
                                        pooling_kernel_size = pooling_kernel_size, 
                                        padding = padding, 
                                        pooling = pooling, 
                                        scale_factor = scale_factor, 
                                        upsample_mode = upsample_mode,
                                        dropout=dropout
                                        ).to(device)
      print("Autoencoder type: Convolutional")
      
    elif extractor_type == 'lstm':
      self.extractor = lstm_autoencoder(in_hidd=autoencoder_dim, 
                                        timesteps=timesteps,
                                        dropout=dropout
                                        ).to(device)
      print("Autoencoder type: LSTM")

    print("Autoencoder Summary:", self.extractor)


    if model_dict != {}:
      self.ensemble = ensemble_model(model_dict, device, heterogeneous=heterogeneous, mode=mode)
    else:
      self.ensemble = NoOpModule()
    # Get current timestamp for the save method
    self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # All the parameters should be tensors and on the same device of the input
    self.register_buffer('mean_X', mean[0].view(1, mean[0].shape[0], 1))  # Mean used for the Standard Scaling approach
    self.register_buffer('mean_Y', mean[1].view(1, 1, mean[1].shape[0]))  # Mean used for the Standard Scaling approach
    self.register_buffer('std_X', std[0].view(1, std[0].shape[0], 1))  # Standard Deviation used for the Standard Scaling approach
    self.register_buffer('std_Y', std[1].view(1, 1, std[1].shape[0]))  # Standard Deviation used for the Standard Scaling approach
    self.register_buffer('max_val_X', max_val[0].view(1, max_val[0].shape[0], 1))  # Maximum Value used for MinMax Scaling approach
    self.register_buffer('max_val_Y', max_val[1].view(1, 1, max_val[1].shape[0]))  # Maximum Value used for MinMax Scaling approach
    self.register_buffer('min_val_X', min_val[0].view(1, min_val[0].shape[0], 1))  # Minimum Value used for MinMax Scaling approach
    self.register_buffer('min_val_Y', min_val[1].view(1, 1, min_val[1].shape[0]))  # Minimum Value used for MinMax Scaling approach

    self.norm = norm # Normalization type ("Std", "MinMax")
  
  def get_models(self):
    if not 'NoOpModule' in str(type(self.ensemble)):
      return self.ensemble.get_models()
    else:
      return [], [], []

  def forward(self, x, y_true):
    
    # Input Transform
    x = x.permute(0,2,1)

    if self.norm == 'Std':
      #print(x.device, self.mean_X.device, self.mean_X)
      
      x = (x - self.mean_X) / (self.std_X + 1e-6) # Standard Scaling (1e-6 prevent division by zero)
      #print(self.mean_Y.shape)
      #print(y_true[:,0,:].unsqueeze(1).shape)
      y_true_norm = (y_true[:,0,:].unsqueeze(1) - self.mean_Y) / (self.std_Y + 1e-6)
    if self.norm == 'MinMax':
      x = (x-self.min_val_X) / (self.max_val_X - self.min_val_X + 1e-6) # MinMax scaling (1e-6 prevent division by zero)
      y_true_norm = (y_true[:,0,:].unsqueeze(1)-self.min_val_Y) / (self.max_val_Y - self.min_val_Y + 1e-6)

    # Extracting Features and sending to the model for output "out"
    merged_z, o = self.extractor(x) # merged_z is the latent space vector to send to the forecaster (ensemble model)
    #print("merged_z:", merged_z.shape)
    # Denormalization for Autoencoder 
    if self.norm == 'Std':
      o = o * self.std_X + self.mean_X # Standard Scaling (1e-6 prevent division by zero)
    if self.norm == 'MinMax':
      o = o * (self.max_val_X - self.min_val_X) + self.min_val_X # MinMax scaling (1e-6 prevent division by zero)

    out = self.ensemble(merged_z, y_true_norm, y_true)
  
    # We return both o (output of the autoencoder to train it) and out (output of the forecaster to train it)
    return out, o
  
  def save(self, autoencoder=False):

    # Create the directory of results
    dir_path = 'results/training_' + self.current_time # path of type 'results/training_2024-12-22_14
    os.makedirs(dir_path, exist_ok=True) # Create the directory

    save_name = 'model.pt' # Model name
    if autoencoder:
      save_name = 'autoencoder.pt'
    save_path = os.path.join(dir_path, save_name) # path of type '/training_2024-12-22_14-57/model.pt
    if autoencoder:
      torch.save(self.extractor.state_dict(), save_path) # Save the model
    else:
      torch.save(self.state_dict(), save_path) # Save the model
    print(f'Model saved to {save_path}')
    return dir_path
  
  def load(self, path, device='cpu', autoencoder=False):
    state_dict = torch.load(path, map_location=torch.device(device))
    if autoencoder:
      self.extractor.load_state_dict(state_dict)
    else:  
      self.load_state_dict(state_dict)
    print("loaded:", path)

'''
if __name__ == '__main__' :

  model_dict = {'mlp': [{'layer_dim_list': [175,40,50,3]}],  
                'ARIMA': [{'p': 2, 'd': 0, 'q': 2, 'ps': 1, 'ds': 1, 'qs': 1, 's': 1}],
                'linear_regressor': [{'in_features': 175, 'out_features': 3}]
                }
  in_kern_out = [[4, 3, 5], [5, 3, 6],[6, 3, 7]]

  if torch.cuda.is_available():
      device = "cuda"
  elif torch.backends.mps.is_available():
      device = "mps"
  else:
      device = "cpu"

  model = complete_model(model_dict, device, in_kern_out=in_kern_out, mode='auto-weighted')
  t = torch.rand(8, 4, 200).to(device) 
  y_true = torch.rand(8, 2, 3).to(device)
  out, o = model(t, y_true)
  '''
 
class NoOpModule(nn.Module):
    def __init__(self):
        super(NoOpModule, self).__init__()

    def forward(self, input1, input2):
        # Do nothing with the inputs
        return input1, input2