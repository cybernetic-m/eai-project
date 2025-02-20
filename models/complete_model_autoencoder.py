import torch.nn as nn
from datetime import datetime
import torch
import os
import sys

modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(modules_path)

from autoencoder import autoencoder
from ensemble_model import ensemble_model

class complete_model_autoencoder(nn.Module):
  def __init__(self, 
               model_dict, 
               device, 
               in_kern_out, 
               pooling_kernel_size = 2, 
               padding = "same", 
               pooling = "max", 
               scale_factor = 2, 
               upsample_mode = "linear", 
               mode='auto-weighted'
               ):
    
    super(complete_model_autoencoder, self).__init__()

    # Define the feature extractor as an autoencoder from which we take
    # the z_merged latent space
    self.extractor = autoencoder(in_kern_out=in_kern_out, 
                                 pooling_kernel_size = pooling_kernel_size, 
                                 padding = padding, 
                                 pooling = pooling, 
                                 scale_factor = scale_factor, 
                                 upsample_mode = upsample_mode
                                 ).to(device)
    
    self.ensemble = ensemble_model(model_dict, device, mode=mode)
    # Get current timestamp for the save method
    self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
  
  def get_models(self):
    return self.ensemble.get_models()

  def forward(self, x, y_true):

    merged_z, o = self.extractor(x) # merged_z is the latent space vector to send to the forecaster (ensemble model)
    out = self.ensemble(merged_z, y_true)

    # We return both o (output of the autoencoder to train it) and out (output of the forecaster to train it)
    return out, o
  
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
 
