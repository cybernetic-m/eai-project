import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

class conv_autoencoder(nn.Module):
   # in_kern_out: it is a list of tuples [[input_channels, kernel_size, output_channels], ...] for each conv layer in the encoder-decoder
   # scale_factor: it is the factor multiplied to the dimension of input at the upsample layer (for example 2 means double the input dimension)
   # padding: you can select the type of padding among "same", "valid", "full"
   # upsample_mode: you can select the type of upsampling among "nearest", "linear"
   # pooling_kernel_size: is the window size of the pooling layer (ex. 2 means halved the dimension)
   # pooling: you can select the type of pooling among "max" (max pooling), "avg" (average pooling)

    def __init__(self, in_kern_out, pooling_kernel_size = 2, padding = "same", pooling = "max", scale_factor = 2, upsample_mode = "linear", dropout = 0.0):
        super(conv_autoencoder, self).__init__()

        self.encoder = conv_encoder(in_kern_out=in_kern_out, 
                               pooling_kernel_size=pooling_kernel_size, 
                               padding=padding, 
                               pooling=pooling,
                               dropout = dropout
                               )
        
        self.decoder = conv_decoder(in_kern_out=[element[::-1] for element in in_kern_out[::-1]], # reverse the list of lists wrt to encoder!
                               scale_factor=scale_factor, 
                               padding=padding, 
                               upsample_mode=upsample_mode
                               )
        
    def forward(self, x):
        #print("x:", x.shape)
        z = self.encoder(x) # Latent space 
        #print("z:", z.shape) 
        merged_z = z.view(z.shape[0], 1,  -1) # Concatenation as [128, 10, 200] -> [128, 1, 200*10]
        #print("merg z:", merged_z.shape)
        o = self.decoder(z)
        #print("o:", o.shape)

        return merged_z, o


class lstm_autoencoder(nn.Module):

   # in_kern_out: you should pass a list of the type [[input_dim, hidden_dim], ...] for each lstm layer
   # dropout: you can choose the amount of dropout

    def __init__(self, in_hidd, dropout = 0.0):
        super(lstm_autoencoder, self).__init__()

        self.encoder = lstm_encoder(in_hidd=in_hidd, 
                                    dropout = dropout
                               )
        
        self.decoder = lstm_decoder(in_hidd=[element[::-1] for element in in_hidd[::-1]], # reverse the list of lists wrt to encoder!
                                    dropout=dropout
                               )
        
    def forward(self, x):
        #print("x:", x.shape)
        z = self.encoder(x) # Latent space 
        #print("z:", z.shape) 
        merged_z = z.view(z.shape[0], 1,  -1) # Concatenation as [128, 200, 100] -> [128, 1, 200*10]
        #print("merg z:", merged_z.shape)
        o = self.decoder(z)
        #print("o:", o.shape)

        return merged_z, o
