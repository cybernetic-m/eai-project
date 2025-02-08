import torch
import torch.nn as nn
import torch.nn.functional as F

class rnn(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1):
    super(rnn, self).__init__()
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)

  def forward(self, x):
    output, hidden_state = self.rnn(x)

    return hidden_state, output
   
class lstm(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1):
    super(lstm, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)

  def forward(self, x):
    output, (hidden_state, cell_state) = self.lstm(x)

    return hidden_state, output
  
class mlp(nn.Module):
  # layer_dim_list: it is a list of layer dimension 
  # (Ex. [20, 30, 50, 3] create a 4 layer MLP with input_dim = 20, hidden_dim_1 = 30 ... until output_dim = 3)
  def __init__(self, layer_dim_list):
    super(mlp, self).__init__()
    self.linear_layers = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1]) for i in range(len(layer_dim_list)-1)])

  def forward(self, x):
    if len(x.shape) > 2:
      #print(x.shape)
      x = x.reshape(x.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    for layer in self.linear_layers:
      #print(x.shape)
      x = layer(x)
      x = F.relu(x)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    return x
  
class linear(nn.Module):
  def __init__(self, in_features, out_features, bias=False):
    super(linear, self).__init__()
    self.linear_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

  def forward(self, x):
    if len(x.shape) > 2:
      x = x.reshape(x.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    x = self.linear_layer(x)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    return x
  
