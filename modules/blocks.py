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
    '''
    if True:
      #self.test = nn.Linear(feature_dim, feature_dim)
      self.test = nn.Sequential(nn.Linear(feature_dim*2, feature_dim))#,nn.Tanh(),nn.Linear(feature_dim*2, feature_dim))
    else:
      self.test = nn.Identity()
    '''
    initialize_weights(self)

  def forward(self, x):
    output, (hidden_state, cell_state) = self.lstm(x)
    hidden_state = hidden_state.permute(1, 0, 2) # [8, 1, 3] [1, 8, 3] Permute beacause rnn, lstm return hidden state [sequence_length, batch_size, features_dim]
    #print(hidden_state.shape)
    #return self.test(hidden_state), self.test(output)
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
  
class cnn(nn.Module):
  # input_dim: number of input features
  # hidden_dim_list: list of hidden layer dimensions
  # output_dim: number of output features
  # kernel_size_list: list of kernel sizes for each convolutional layer
  def __init__(self, input_dim, hidden_dim_list, output_dim, kernel_size_list):
    super(cnn, self).__init__()
    self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, hidden_dim_list[0], kernel_size=kernel_size_list[0])] + 
                                    [nn.Conv1d(hidden_dim_list[i], hidden_dim_list[i+1], kernel_size=kernel_size_list[i+1]) for i in range(len(hidden_dim_list)-1)])
    self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim_list[-1], output_dim)])
    initialize_weights(self)

  def forward(self, x):
    if len(x.shape) == 2:
      x = x.unsqueeze(1) # Adding a dimension because the input is [8, 20] -> [8, 1, 20]
    x = x.transpose(1, 2) # Transpose to [8, 20, 1] for Conv1d
    for layer_i in range(len(self.conv_layers)):
      x = self.conv_layers[layer_i](x)
      x = F.tanh(x)
    x = x.view(x.size(0), -1) # Flatten
    x = self.linear_layers[0](x)
    x = x.unsqueeze(1) # Adding a dimension because of the previous flattening [8, 3] -> [8,1,3]
    #print(x.shape)
    return x

class encoder(nn.Module):
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
  
class decoder(nn.Module):
# in_kern_out: it is a list of lists [[input_channels, kernel_size, output_channels], ...] for each conv layer in the decoder   
# scale_factor: it is the factor multiplied to the dimension of input at the upsample layer (for example 2 means double the input dimension)
#  padding: you can select the type of padding among "same", "valid", "full"
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
    for i, (conv, upsample) in enumerate(zip(self.conv_layers, self.upsample_layers)):
      if i != len(self.conv_layers):
        x = conv(x)
        x = F.leaky_relu(x)
        x = upsample(x)
      else:
        x = conv(x)
        #x = F.sigmoid(x)
        x = upsample(x)
    return x

   
  
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