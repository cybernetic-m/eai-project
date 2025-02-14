import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

class rnn(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1):
    super(rnn, self).__init__()
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)
    initialize_weights(self)

  def forward(self, x):
    output, hidden_state = self.rnn(x)
    hidden_state = hidden_state.permute(1, 2, 0)
    return hidden_state, output
   
class lstm(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1):
    super(lstm, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)
    self.hidden_norm = nn.BatchNorm1d(num_features=feature_dim)
    initialize_weights(self)

  def forward(self, x):
    output, (hidden_state, cell_state) = self.lstm(x)
    hidden_state = hidden_state.permute(1, 2, 0) # [8, 1, 3] [1, 8, 3] Permute beacause rnn, lstm return hidden state [sequence_length, batch_size, features_dim]
    return hidden_state, output
  
class mlp(nn.Module):
  # layer_dim_list: it is a list of layer dimension 
  # (Ex. [20, 30, 50, 3] create a 4 layer MLP with input_dim = 20, hidden_dim_1 = 30 ... until output_dim = 3)
  def __init__(self, layer_dim_list):
    super(mlp, self).__init__()
    #print(layer_dim_list)
    self.linear_layers = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1]) for i in range(len(layer_dim_list)-1)])
    initialize_weights(self)

  def forward(self, x):
    if len(x.shape) > 2:
      #print(x.shape)
      x = x.reshape(x.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    for layer_i in range(len(self.linear_layers)-1):
      #print(x.shape)
      x = self.linear_layers[layer_i](x)
      x = F.tanh(x)
    x = self.linear_layers[-1](x)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    return x
  
class linear(nn.Module):
  def __init__(self, in_features, out_features, bias=False):
    super(linear, self).__init__()
    self.linear_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    initialize_weights(self)

  def forward(self, x):
    if len(x.shape) > 2:
      x = x.reshape(x.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    x = self.linear_layer(x)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    return x
  
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias with zeros
        elif isinstance(m, nn.RNN) or isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    init.xavier_uniform_(param)  # Xavier for input-hidden weights
                elif "weight_hh" in name:
                    init.orthogonal_(param)  # Orthogonal for hidden-hidden weights
                elif "bias" in name:
                    init.zeros_(param)  # Initialize bias with zeros
        elif isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)  # Set batch norm weights to 1
            init.zeros_(m.bias)   # Set batch norm bias to 0