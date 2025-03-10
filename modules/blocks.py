import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

class rnn(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1):
    super(rnn, self).__init__()
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)
    self.input_dim = input_dim
    initialize_weights(self)

  def forward(self, x):
    if x.shape[-1] != self.input_dim:
      x = x.permute(0,2,1)
    #print("rnn x:", x.shape)
    output, hidden_state = self.rnn(x)
    #print("rnn out:", output.shape)
    return output
   
class lstm(nn.Module):
  def __init__(self, feature_dim, input_dim, num_layers=1, dropout=0.0):
    super(lstm, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    self.input_dim = input_dim
    
    initialize_weights(self)

  def forward(self, x):
    #print("HERE:", x.shape)
    if x.shape[-1] != self.input_dim:
      x = x.permute(0,2,1)
    #print("input lstm:", x.shape)
    output, (hidden_state, cell_state) = self.lstm(x)
    #print("output_lstm:", output.shape)
    return output
  
class mlp(nn.Module):
  # layer_dim_list: it is a list of layer dimension 
  # (Ex. [20, 30, 50, 3] create a 4 layer MLP with input_dim = 20, hidden_dim_1 = 30 ... until output_dim = 3)
  def __init__(self, layer_dim_list, bias=True):
    super(mlp, self).__init__()
    #print(layer_dim_list)
    self.linear_layers = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1], bias=bias) for i in range(len(layer_dim_list)-1)])
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
    #print("x linear:", len(x))
    if len(x.shape) > 2:
      x = x.reshape(x.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    #print("x linear reshaped:", x.shape)
    x = self.linear_layer(x)
    #print("x after linear:", x.shape)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    return x

class rnn_regressor(nn.Module):
  def __init__(self, feature_dim, input_dim, out_features, bias=False, num_layers=1):
    super(rnn_regressor, self).__init__()
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True)
    self.linear = nn.Linear(in_features=feature_dim, out_features=out_features, bias=bias)
    self.input_dim = input_dim
    initialize_weights(self)

  def forward(self, x):
    if x.shape[-1] != self.input_dim:
      x = x.permute(0,2,1)
    #print("input lstm:", x.shape)
    rnn_out, hidden_state = self.rnn(x)
    if len(rnn_out.shape) > 2:
      rnn_out = rnn_out.reshape(rnn_out.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    #print("x linear reshaped:", x.shape)
    out = self.linear(rnn_out)
    #print("x after linear:", x.shape)
    out = out.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    #print("output_lstm:", output.shape)
    return out
  
class lstm_regressor(nn.Module):
  def __init__(self, feature_dim, input_dim, out_features, num_layers=1, dropout=0.0, bias=False):
    super(lstm_regressor, self).__init__()
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
    self.linear = nn.Linear(in_features=feature_dim, out_features=out_features, bias=bias)
    self.input_dim = input_dim
    
    initialize_weights(self)

  def forward(self, x):
    if x.shape[-1] != self.input_dim:
      x = x.permute(0,2,1)
    #print("input lstm:", x.shape)
    lstm_out, (hidden_state, cell_state) = self.lstm(x)
    if len(lstm_out.shape) > 2:
      lstm_out = lstm_out.reshape(lstm_out.shape[0], -1) # Flatten in case in which the tensor in input is [8, 5, 3] -> [8, 20]
    #print("x linear reshaped:", x.shape)
    out = self.linear(lstm_out)
    #print("x after linear:", x.shape)
    out = out.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    #print("output_lstm:", output.shape)
    return out

class conv_encoder(nn.Module):
   # in_kern_out: it is a list of lists [[input_channels, kernel_size, output_channels], ...] for each conv layer in the encoder
   # pooling_kernel_size: is the window size of the pooling layer (ex. 2 means halved the dimension)
   # padding: you can select the type of padding among "same", "valid", "full"
   # pooling: you can select the type of pooling among "max" (max pooling), "avg" (average pooling)
  def __init__(self, in_kern_out, pooling_kernel_size = 2, padding = "same", pooling = "max", dropout=0.0):
    super().__init__()

    self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=in_channels, 
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding = padding
                                                ) 
                                                for in_channels, kernel_size, out_channels in in_kern_out])
    
    if pooling == 'max':
      self.pooling_layers = nn.ModuleList([nn.MaxPool1d(kernel_size=pooling_kernel_size) for _ in in_kern_out])
    if pooling == 'avg':
      self.pooling_layers = nn.ModuleList([nn.AvgPool1d(kernel_size=pooling_kernel_size) for _ in in_kern_out])
        
    self.norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=out_channels) 
                                                for _, _, out_channels in in_kern_out])

    self.dropout = dropout
    
    initialize_weights(self)

  def forward(self, x):
    for conv, pooling, norm in zip(self.conv_layers, self.pooling_layers, self.norm_layers):
       x = conv(x)
       x = norm(x)
       x = F.leaky_relu(x)
       x = pooling(x)
       x = F.dropout(x, p=self.dropout)
    return x
    
class conv_decoder(nn.Module):
# in_kern_out: it is a list of lists [[input_channels, kernel_size, output_channels], ...] for each conv layer in the decoder   
# scale_factor: it is the factor multiplied to the dimension of input at the upsample layer (for example 2 means double the input dimension)
# padding: you can select the type of padding among "same", "valid", "full"
# upsample_mode: you can select the type of upsampling among "nearest", "linear"

  def __init__(self, in_kern_out, scale_factor = 2, padding = "same", upsample_mode = "linear"):
    super().__init__()

    self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=in_channels, 
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                padding = padding
                                                ) 
                                                for in_channels, kernel_size, out_channels in in_kern_out])
    
    self.upsample_layers = nn.ModuleList([nn.Upsample(scale_factor=scale_factor, mode=upsample_mode) for _ in in_kern_out])
    initialize_weights(self)

  def forward(self, x):
    for conv, upsample in zip(self.conv_layers, self.upsample_layers):
      x = conv(x)
      x = F.leaky_relu(x)
      x = upsample(x)

    return x
  
class lstm_encoder(nn.Module):
   # in_hidd: you should pass a list of the type [[input_dim, hidden_dim], ...] for each lstm layer
   # dropout : you can choose the amount of dropout

  def __init__(self, in_hidd, num_layers, dropout = 0.0):
    super().__init__()

    self.lstm_layers = nn.ModuleList([lstm( 
                                      input_dim = input_dim,
                                      feature_dim = hidden_dim, 
                                      num_layers=num_layers,
                                      dropout=dropout
                                      ) for input_dim, hidden_dim in in_hidd])
        
    #self.norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=out_channels) 
    #                                            for _, _, out_channels in in_kern_out])
    
    initialize_weights(self)

  def forward(self, x):
    for lstm in self.lstm_layers:
       x = lstm(x) 
    return x
  
class lstm_decoder(nn.Module):
   # in_hidd: you should pass a list of the type [[input_dim, hidden_dim], ...] for each lstm layer
   # dropout : you can choose the amount of dropout

  def __init__(self, in_hidd, timesteps, num_layers, dropout = 0.0):
    super().__init__()

    self.lstm_layers = nn.ModuleList([lstm( 
                                      input_dim = input_dim,
                                      feature_dim = hidden_dim, 
                                      num_layers=num_layers,
                                      dropout=dropout
                                      ) for input_dim, hidden_dim in in_hidd])
            
    #self.norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=out_channels) 
    #                                            for _, _, out_channels in in_kern_out])
    self.linear_decoder = nn.Linear(in_features=in_hidd[-1][-1]*timesteps, out_features=in_hidd[-1][-1]*timesteps)
    
    initialize_weights(self)

  def forward(self, x):
    for lstm in self.lstm_layers:
       x = lstm(x)
    x_flatten = x.reshape(x.shape[0], 1, -1) # Flatten [256, 4, 200] -> [256, 1, 800]
    o = self.linear_decoder(x_flatten)
    o = o.view(o.shape[0], x.shape[2], x.shape[1]) # Reshaping to return to the input dimensions [batch_size, feature_dim, seq_len] ([256, 4, 200])

    return o

   
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)  # Xavier initialization for linear layers
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias with zeros
        elif isinstance(m, nn.RNN) or isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    init.xavier_normal_(param)  # Xavier for input-hidden weights
                elif "weight_hh" in name:
                    init.orthogonal_(param)  # Orthogonal for hidden-hidden weights
                elif "bias" in name:
                    init.zeros_(param)  # Initialize bias with zeros
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)  # Set batch norm weights to 1
            init.zeros_(m.bias)   # Set batch norm bias to 0
        elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)  # Kaiming initialization for convolutional layers
            if m.bias is not None:
                init.zeros_(m.bias)  # Initialize bias with zeros