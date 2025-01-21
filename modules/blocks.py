import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn(nn.Module):
  def __init__(self, feature_dim, input_dim):
    super(rnn, self).__init__()
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=feature_dim, batch_first=True)

  def forward(self, x):
    output, (hidden_state, cell_state) = self.rnn(x)

    return output
  
class lstm(nn.Module):
  def __init__(self, feature_dim, input_dim):
    super(lstm, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, batch_first=True)

  def forward(self, x):
    output, (hidden_state, cell_state) = self.lstm(x)
    return output
  
class mlp(nn.Module):
  # layer_dim_list: it is a list of layer dimension 
  # (Ex. [20, 30, 50, 3] create a 4 layer MLP with input_dim = 20, hidden_dim_1 = 30 ... until output_dim = 3)
  def __init__(self, layer_dim_list):
    super(mlp, self).__init__()
    self.linear_layers = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1]) for i in range(len(layer_dim_list)-1)])

  def forward(self, x):
    for layer in self.linear_layers:
      x = layer(x)
      x = F.relu(x)
    return x
